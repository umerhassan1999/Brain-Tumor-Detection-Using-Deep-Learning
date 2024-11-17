import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QMainWindow
from PyQt5.QtGui import QPixmap
import numpy as np
import cv2
import sys

import os
import torch
from numpy import *
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import PySimpleGUI as sg


class Ui_MainWindow(object):
    imagePath = ' '

    ## Main Window Functins ##
    def imgupload(self):
            fname, _filter = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image File", '.', "(*.jpg)")
            self.imagePath = fname


            if self.imagePath == '':
                    (self.label_9.setStyleSheet("background-color: red;"))

            else:
                    (self.label_9.setStyleSheet("background-color: #85ff77;"))
                    (self.label_8.setStyleSheet("background-color: #85ff77;"))
                    (self.label_13.setStyleSheet("background-color: #85ff77;"))
                    self.mupload.setPixmap(QPixmap(self.imagePath))
                    self.pupload.setPixmap(QPixmap(self.imagePath))
                    self.cupload.setPixmap(QPixmap(self.imagePath))

                    img = cv2.imread(self.imagePath)
                    cv2.imwrite('Original_Pic.jpg', img)
    def imgdelete(self):
            if self.imagePath == " ":
                    ((self.label_9.setStyleSheet("background-color:red;")))
            else:
                #if os.path.isfile("Original_Pic.jpg"):
                os.remove("Original_Pic.jpg")
                self.imagePath = " "
                self.mupload.setText(" ")
                self.pupload.setText(" ")
                self.cupload.setText(" ")
                self.label_9.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
                self.label_8.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
                self.label_13.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
                   #self.imagePath == ' '
                #else:



    ## Pre-Processing Functions ##
    def imgresize(self):
            if self.imagePath == " ":
                    ((self.label_24.setStyleSheet("background-color:red;")))
            else:
                    self.label_24.setStyleSheet("background-color: #85ff77;")
                    img = cv2.imread(self.imagePath)
                    res = cv2.resize(img, (512, 512))
                    cv2.imwrite(self.imagePath, res)
                    self.resize.setPixmap(QPixmap(self.imagePath))
    def imggraycolor(self):
            if self.imagePath == " ":
                    ((self.label_25.setStyleSheet("background-color:red;")))
            else:
                    self.label_25.setStyleSheet("background-color: #85ff77;")

                    image = cv2.imread(self.imagePath)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(self.imagePath, gray)
                    self.gray.setPixmap(QPixmap(self.imagePath))
    def imgenhancement(self):
            if self.imagePath == " ":
                    ((self.label_26.setStyleSheet("background-color:red;")))
            else:
                    self.label_26.setStyleSheet("background-color: #85ff77;")
                    img = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
                    norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    norm_img2 = (255 * norm_img1).astype(np.uint8)
                    cv2.imwrite(self.imagePath, norm_img2)
                    self.contrst.setPixmap(QPixmap(self.imagePath))
    def imgmedian(self):
            if self.imagePath == " ":
                    ((self.label_28.setStyleSheet("background-color:red;")))
            else:
                    self.label_28.setStyleSheet("background-color: #85ff77;")
                    img = cv2.imread(self.imagePath)
                    median = cv2.medianBlur(img, 5)
                    cv2.imwrite(self.imagePath, median)
                    self.median.setPixmap(QPixmap(self.imagePath))
    def imgclear(self):
            self.label_8.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
            self.label_24.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
            self.label_25.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
            self.label_26.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
            self.label_28.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
            self.resize.setText(" ")
            self.gray.setText(" ")
            self.contrst.setText(" ")
            self.median.setText(" ")
            self.pupload.setText(" ")

    ## Results Functions ##

    def prediction(self):
        if self.imagePath== " ":
             self.cupload.setPixmap(QPixmap("pic/error.png"))

        else:
             self.cresult.setPixmap(QPixmap(self.imagePath))
             (self.label_14.setStyleSheet("background-color: #85ff77;"))
             classes = ('Acoustic_neuroma', 'Glioma', 'Meningioma', 'No_Tumor', 'Pituitary')
             transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

             def image_loader(image_name):
                 """load image, returns tensor"""
                 image = Image.open(image_name).convert('RGB')  # load single image
                 image = transform(image).float()  # apply transformation
                 image = Variable(image, requires_grad=True)  # Convert it to tensor
                 image = image.unsqueeze(0)
                 return image

             class CNN(nn.Module):
                 def __init__(self):
                     super(CNN, self).__init__()
                     self.conv1 = nn.Conv2d(3, 32, 3)
                     self.pool = nn.MaxPool2d(2, 2)
                     self.conv2 = nn.Conv2d(32, 64, 3)
                     self.fc1 = nn.Linear(64 * 14 * 14, 40)
                     self.fc2 = nn.Linear(40, 5)

                 # self.softmax = nn.Softmax(dim=1)
                 def forward(self, x):
                     x = self.pool(F.relu(self.conv1(x)))
                     x = self.pool(F.relu(self.conv2(x)))
                     x = x.view(-1, 64 * 14 * 14)
                     x = self.fc1(x)
                     x = F.relu(x)
                     x = self.fc2(x)
                     # x = self.softmax(x)
                     return x

             model = CNN()
             # print(model)
             state_dict = torch.load('Model/brainmodel.pth')['state_dict']
             model.load_state_dict(state_dict)
             # change this path according to your iamge path
             image = image_loader(self.imagePath)
             output = model(image)
             # print(output.data.cpu().numpy()) #HIGHEST CONFIDENCE
             _, predicted = torch.max(output, 1)
             w = classes[predicted]
             # w =str(w)


             self.label_45.setText(w)
    def imgclear2(self):
         if os.path.isfile("Original_Pic.jpg"):
            os.remove("Original_Pic.jpg")
         else:
          self.imagePath = " "

         self.cupload.setText(" ")
         self.cresult.setText(" ")
         self.label_14.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")
         self.label_13.setStyleSheet("background-color: rgb(74, 98, 116);\n""color: rgb(234, 255, 255);")

         self.label_45.setText(" ")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1030, 663)
        MainWindow.setMaximumSize(QtCore.QSize(1300, 750))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("pics/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 100, 1031, 561))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(20)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet("QWidget {background-color: rgb(212, 220, 169); color: black;}\n"
"\n"
"\n"
"\n"
"QTabBar::tab:selected{background-color: rgb(74, 98, 116) ; color:#bdc4c9; }\n"
"\n"
"QTabBar::tab:hover{background-color: rgb(74, 98, 116);}")
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.South)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setObjectName("tabWidget")
        self.main = QtWidgets.QWidget()
        self.main.setObjectName("main")
        self.label_2 = QtWidgets.QLabel(self.main)
        self.label_2.setGeometry(QtCore.QRect(410, 0, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_2.setLineWidth(3)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.main)
        self.label_3.setGeometry(QtCore.QRect(80, 70, 461, 381))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("pics/7.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.mupload = QtWidgets.QLabel(self.main)
        self.mupload.setGeometry(QtCore.QRect(700, 110, 271, 271))
        self.mupload.setFrameShape(QtWidgets.QFrame.Box)
        self.mupload.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.mupload.setLineWidth(2)
        self.mupload.setText("")
        self.mupload.setScaledContents(True)
        self.mupload.setObjectName("mupload")
        self.btnupload = QtWidgets.QPushButton(self.main)
        self.btnupload.setGeometry(QtCore.QRect(700, 400, 271, 41))
        self.btnupload.setStyleSheet("QPushButton { background-color: rgb(74, 98, 116);\n"
"color: rgb(234, 255, 255); border-radius:5px;}\n"
"\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btnupload.setObjectName("btnupload")
        self.btnupload.clicked.connect(self.imgupload)

        self.btnmclr = QtWidgets.QPushButton(self.main)
        self.btnmclr.setGeometry(QtCore.QRect(1040, 470, 171, 41))
        self.btnmclr.setStyleSheet("QPushButton { color: rgb(234, 255, 255);background-color: rgb(74, 98, 116); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btnmclr.setObjectName("btnmclr")
        self.btnmdelete = QtWidgets.QPushButton(self.main)
        self.btnmdelete.setGeometry(QtCore.QRect(700, 460, 271, 41))
        self.btnmdelete.setStyleSheet("QPushButton {background-color: rgb(74, 98, 116);\n"
"color: rgb(234, 255, 255); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btnmdelete.setObjectName("btnmdelete")
        self.btnmdelete.clicked.connect(self.imgdelete)

        self.label_9 = QtWidgets.QLabel(self.main)
        self.label_9.setGeometry(QtCore.QRect(700, 70, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("background-color: rgb(74, 98, 116);\n"
"color: rgb(234, 255, 255);")
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_9.setLineWidth(2)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.tabWidget.addTab(self.main, "")
        self.pre = QtWidgets.QWidget()
        self.pre.setObjectName("pre")
        self.pupload = QtWidgets.QLabel(self.pre)
        self.pupload.setGeometry(QtCore.QRect(300, 160, 190, 190))
        self.pupload.setFrameShape(QtWidgets.QFrame.Box)
        self.pupload.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.pupload.setLineWidth(2)
        self.pupload.setText("")
        self.pupload.setScaledContents(True)
        self.pupload.setObjectName("pupload")
        self.groupBox = QtWidgets.QGroupBox(self.pre)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 241, 601))
        self.groupBox.setStyleSheet("color: rgb(234, 255, 255);background-color: rgb(88, 90, 86);")
        self.groupBox.setTitle("")
        self.groupBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setChecked(False)
        self.groupBox.setObjectName("groupBox")
        self.btnresize = QtWidgets.QPushButton(self.groupBox)
        self.btnresize.setGeometry(QtCore.QRect(30, 80, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.btnresize.setFont(font)
        self.btnresize.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btnresize.setObjectName("btnresize")
        self.btnresize.clicked.connect(self.imgresize)

        self.btngray = QtWidgets.QPushButton(self.groupBox)
        self.btngray.setGeometry(QtCore.QRect(30, 150, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.btngray.setFont(font)
        self.btngray.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btngray.setObjectName("btngray")
        self.btngray.clicked.connect(self.imggraycolor)

        self.btncontrst = QtWidgets.QPushButton(self.groupBox)
        self.btncontrst.setGeometry(QtCore.QRect(30, 220, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.btncontrst.setFont(font)
        self.btncontrst.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btncontrst.setObjectName("btncontrst")
        self.btncontrst.clicked.connect(self.imgenhancement)

        self.btnmedian = QtWidgets.QPushButton(self.groupBox)
        self.btnmedian.setGeometry(QtCore.QRect(30, 290, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.btnmedian.setFont(font)
        self.btnmedian.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btnmedian.setObjectName("btnmedian")
        self.btnmedian.clicked.connect(self.imgmedian)

        self.btnpclear = QtWidgets.QPushButton(self.groupBox)
        self.btnpclear.setGeometry(QtCore.QRect(30, 360, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.btnpclear.setFont(font)
        self.btnpclear.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.btnpclear.setObjectName("btnpclear")
        self.btnpclear.clicked.connect(self.imgclear)

        self.resize = QtWidgets.QLabel(self.pre)
        self.resize.setGeometry(QtCore.QRect(540, 60, 190, 190))
        self.resize.setFrameShape(QtWidgets.QFrame.Box)
        self.resize.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.resize.setLineWidth(2)
        self.resize.setText("")
        self.resize.setScaledContents(True)
        self.resize.setObjectName("resize")
        self.gray = QtWidgets.QLabel(self.pre)
        self.gray.setGeometry(QtCore.QRect(770, 60, 190, 190))
        self.gray.setFrameShape(QtWidgets.QFrame.Box)
        self.gray.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.gray.setLineWidth(2)
        self.gray.setText("")
        self.gray.setScaledContents(True)
        self.gray.setObjectName("gray")
        self.contrst = QtWidgets.QLabel(self.pre)
        self.contrst.setGeometry(QtCore.QRect(540, 300, 190, 190))
        self.contrst.setFrameShape(QtWidgets.QFrame.Box)
        self.contrst.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.contrst.setLineWidth(2)
        self.contrst.setText("")
        self.contrst.setScaledContents(True)
        self.contrst.setObjectName("contrst")
        self.median = QtWidgets.QLabel(self.pre)
        self.median.setGeometry(QtCore.QRect(770, 300, 190, 190))
        self.median.setFrameShape(QtWidgets.QFrame.Box)
        self.median.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.median.setLineWidth(2)
        self.median.setText("")
        self.median.setScaledContents(True)
        self.median.setObjectName("median")
        self.label_8 = QtWidgets.QLabel(self.pre)
        self.label_8.setGeometry(QtCore.QRect(300, 350, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color: rgb(74, 98, 116);\n"
"color: rgb(234, 255, 255);")
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_8.setLineWidth(2)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_24 = QtWidgets.QLabel(self.pre)
        self.label_24.setGeometry(QtCore.QRect(540, 250, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_24.setFont(font)
        self.label_24.setStyleSheet("background-color: rgb(74, 98, 116); color: rgb(234, 255, 255);")
        self.label_24.setFrameShape(QtWidgets.QFrame.Box)
        self.label_24.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_24.setLineWidth(2)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.pre)
        self.label_25.setGeometry(QtCore.QRect(770, 250, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_25.setFont(font)
        self.label_25.setStyleSheet("background-color: rgb(74, 98, 116); color: rgb(234, 255, 255);")
        self.label_25.setFrameShape(QtWidgets.QFrame.Box)
        self.label_25.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_25.setLineWidth(2)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.pre)
        self.label_26.setGeometry(QtCore.QRect(540, 490, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_26.setFont(font)
        self.label_26.setStyleSheet("background-color: rgb(74, 98, 116); color: rgb(234, 255, 255);")
        self.label_26.setFrameShape(QtWidgets.QFrame.Box)
        self.label_26.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_26.setLineWidth(2)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.label_28 = QtWidgets.QLabel(self.pre)
        self.label_28.setGeometry(QtCore.QRect(770, 490, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_28.setFont(font)
        self.label_28.setStyleSheet("background-color: rgb(74, 98, 116); color: rgb(234, 255, 255);")
        self.label_28.setFrameShape(QtWidgets.QFrame.Box)
        self.label_28.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_28.setLineWidth(2)
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")
        self.label_6 = QtWidgets.QLabel(self.pre)
        self.label_6.setGeometry(QtCore.QRect(520, 0, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_6.setLineWidth(3)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.tabWidget.addTab(self.pre, "")
        self.clasi = QtWidgets.QWidget()
        self.clasi.setObjectName("clasi")
        self.groupBox_4 = QtWidgets.QGroupBox(self.clasi)
        self.groupBox_4.setGeometry(QtCore.QRect(0, 0, 241, 531))
        self.groupBox_4.setStyleSheet("color: rgb(234, 255, 255); background-color: rgb(88, 90, 86);")
        self.groupBox_4.setTitle("")
        self.groupBox_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.groupBox_4.setFlat(False)
        self.groupBox_4.setCheckable(False)
        self.groupBox_4.setChecked(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_28 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_28.setGeometry(QtCore.QRect(28, 190, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.pushButton_28.setFont(font)
        self.pushButton_28.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.pushButton_28.setObjectName("pushButton_28")
        self.pushButton_28.clicked.connect(self.prediction)
        self.pushButton_30 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_30.setGeometry(QtCore.QRect(30, 260, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.pushButton_30.setFont(font)
        self.pushButton_30.setStyleSheet("QPushButton {color:black; background-color: rgb(212, 220, 169); border-radius:5px;}\n"
"QPushButton:pressed { background-color: red; border-radius:5px; color: rgb(205, 223, 230);}")
        self.pushButton_30.setObjectName("pushButton_30")
        self.pushButton_30.clicked.connect(self.imgclear2)
        self.cupload = QtWidgets.QLabel(self.clasi)
        self.cupload.setGeometry(QtCore.QRect(370, 100, 250, 250))
        self.cupload.setFrameShape(QtWidgets.QFrame.Box)
        self.cupload.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.cupload.setLineWidth(2)
        self.cupload.setText("")
        self.cupload.setScaledContents(True)
        self.cupload.setObjectName("cupload")
        self.cresult = QtWidgets.QLabel(self.clasi)
        self.cresult.setGeometry(QtCore.QRect(680, 100, 250, 250))
        self.cresult.setFrameShape(QtWidgets.QFrame.Box)
        self.cresult.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.cresult.setLineWidth(2)
        self.cresult.setText("")
        self.cresult.setScaledContents(True)
        self.cresult.setObjectName("cresult")
        self.label_13 = QtWidgets.QLabel(self.clasi)
        self.label_13.setGeometry(QtCore.QRect(370, 350, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("background-color: rgb(74, 98, 116);\n"
"color: rgb(234, 255, 255);")
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_13.setLineWidth(2)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.clasi)
        self.label_14.setGeometry(QtCore.QRect(680, 350, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(14)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("background-color: rgb(74, 98, 116);\n"
"color: rgb(234, 255, 255);")
        self.label_14.setFrameShape(QtWidgets.QFrame.Box)
        self.label_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_14.setLineWidth(2)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.clasi)
        self.label_15.setGeometry(QtCore.QRect(520, 0, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_15.setFrameShape(QtWidgets.QFrame.Box)
        self.label_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_15.setLineWidth(3)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.label_45 = QtWidgets.QLabel(self.clasi)
        self.label_45.setGeometry(QtCore.QRect(590, 430, 191, 31))
        self.label_45.setFrameShape(QtWidgets.QFrame.Box)
        self.label_45.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_45.setLineWidth(2)
        self.label_45.setText("")
        self.label_45.setObjectName("label_45")
        self.label_46 = QtWidgets.QLabel(self.clasi)
        self.label_46.setGeometry(QtCore.QRect(440, 430, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_46.setFont(font)
        self.label_46.setAlignment(QtCore.Qt.AlignCenter)
        self.label_46.setObjectName("label_46")
        self.tabWidget.addTab(self.clasi, "")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1032, 101))
        font = QtGui.QFont()
        font.setFamily("Microsoft Himalaya")
        font.setPointSize(28)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(234, 255, 255); \n"
"background-color: rgb(88, 90, 86);")
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setLineWidth(4)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 111, 81))
        self.label_4.setStyleSheet("background-color: rgb(88, 90, 86);")
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("pics/2.png"))
        self.label_4.setScaledContents(True)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Main Window"))
        self.btnupload.setText(_translate("MainWindow", "Upload Image"))
        self.btnmclr.setText(_translate("MainWindow", "Clear"))
        self.btnmdelete.setText(_translate("MainWindow", "Delete Image"))
        self.label_9.setText(_translate("MainWindow", "Input Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.main), _translate("MainWindow", "                   Main Window                    "))
        self.btnresize.setText(_translate("MainWindow", "Image Resize"))
        self.btngray.setText(_translate("MainWindow", "Gray Scale"))
        self.btncontrst.setText(_translate("MainWindow", "Image Enhancement"))
        self.btnmedian.setText(_translate("MainWindow", "Median Filter"))
        self.btnpclear.setText(_translate("MainWindow", "Clear All"))
        self.label_8.setText(_translate("MainWindow", "Orignial Image"))
        self.label_24.setText(_translate("MainWindow", "Image Resize"))
        self.label_25.setText(_translate("MainWindow", "Gray Scale"))
        self.label_26.setText(_translate("MainWindow", "Normalize"))
        self.label_28.setText(_translate("MainWindow", "Median Filter"))
        self.label_6.setText(_translate("MainWindow", "Pre-Processing"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.pre), _translate("MainWindow", "                      Pre-Processing                     "))
        self.pushButton_28.setText(_translate("MainWindow", "Prediction"))
        self.pushButton_30.setText(_translate("MainWindow", "Clear All"))
        self.label_13.setText(_translate("MainWindow", "Orignial Image"))
        self.label_14.setText(_translate("MainWindow", "Result Image"))
        self.label_15.setText(_translate("MainWindow", "Classifier"))
        self.label_46.setText(_translate("MainWindow", "Tumor Type"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.clasi), _translate("MainWindow", "                        Prediction                            "))
        self.label.setText(_translate("MainWindow", "Automated Brain Tumor Classification"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    IMAGE_FILE = r'pics/2.png'
    time = 2000

    sg.Window('Window Title', [[sg.Image(IMAGE_FILE)]], transparent_color=sg.theme_background_color(), no_titlebar=True,
              keep_on_top=True).read(timeout=time, close=True)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
