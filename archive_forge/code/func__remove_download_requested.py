import sys
from bookmarkwidget import BookmarkWidget
from browsertabwidget import BrowserTabWidget
from downloadwidget import DownloadWidget
from findtoolbar import FindToolBar
from webengineview import QWebEnginePage, WebEngineView
from PySide2 import QtCore
from PySide2.QtCore import Qt, QUrl
from PySide2.QtGui import QCloseEvent, QKeySequence, QIcon
from PySide2.QtWidgets import (qApp, QAction, QApplication, QDesktopWidget,
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem, QWebEnginePage,
def _remove_download_requested(self):
    download_widget = self.sender()
    self.statusBar().removeWidget(download_widget)
    del download_widget