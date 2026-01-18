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
def _add_tool_bar_bookmark(self):
    index = self._tab_widget.currentIndex()
    if index >= 0:
        url = self._tab_widget.url()
        title = self._tab_widget.tabText(index)
        icon = self._tab_widget.tabIcon(index)
        self._bookmark_widget.add_tool_bar_bookmark(url, title, icon)