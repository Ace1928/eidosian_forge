from functools import partial
import sys
from bookmarkwidget import BookmarkWidget
from webengineview import WebEngineView
from historywindow import HistoryWindow
from PySide2 import QtCore
from PySide2.QtCore import QPoint, Qt, QUrl
from PySide2.QtWidgets import (QAction, QMenu, QTabBar, QTabWidget)
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem,
def _index_of_page(self, web_page):
    for p in range(0, len(self._webengineviews)):
        if self._webengineviews[p].page() == web_page:
            return p
    return -1