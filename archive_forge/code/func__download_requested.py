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
def _download_requested(self, item):
    for old_download in self.statusBar().children():
        if type(old_download).__name__ == 'download_widget' and old_download.state() != QWebEngineDownloadItem.DownloadInProgress:
            self.statusBar().removeWidget(old_download)
            del old_download
    item.accept()
    download_widget = download_widget(item)
    download_widget.removeRequested.connect(self._remove_download_requested, Qt.QueuedConnection)
    self.statusBar().addWidget(download_widget)