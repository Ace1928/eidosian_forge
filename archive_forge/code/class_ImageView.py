import os, sys
from PySide2.QtCore import QDate, QDir, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QClipboard, QGuiApplication, QDesktopServices, QIcon
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import (QAction, qApp, QApplication, QHBoxLayout, QLabel,
from PySide2.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo
from PySide2.QtMultimediaWidgets import QCameraViewfinder
class ImageView(QWidget):

    def __init__(self, previewImage, fileName):
        super(ImageView, self).__init__()
        self.fileName = fileName
        mainLayout = QVBoxLayout(self)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap.fromImage(previewImage))
        mainLayout.addWidget(self.imageLabel)
        topLayout = QHBoxLayout()
        self.fileNameLabel = QLabel(QDir.toNativeSeparators(fileName))
        self.fileNameLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        topLayout.addWidget(self.fileNameLabel)
        topLayout.addStretch()
        copyButton = QPushButton('Copy')
        copyButton.setToolTip('Copy file name to clipboard')
        topLayout.addWidget(copyButton)
        copyButton.clicked.connect(self.copy)
        launchButton = QPushButton('Launch')
        launchButton.setToolTip('Launch image viewer')
        topLayout.addWidget(launchButton)
        launchButton.clicked.connect(self.launch)
        mainLayout.addLayout(topLayout)

    def copy(self):
        QGuiApplication.clipboard().setText(self.fileNameLabel.text())

    def launch(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.fileName))