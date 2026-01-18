import sys
from PySide2 import QtCore
from PySide2.QtCore import QDir, QFileInfo, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QDesktopServices
from PySide2.QtWidgets import (QAction, QLabel, QMenu, QProgressBar,
from PySide2.QtWebEngineWidgets import QWebEngineDownloadItem
Lets you track progress of a QWebEngineDownloadItem.