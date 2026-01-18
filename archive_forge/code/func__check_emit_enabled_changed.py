from functools import partial
import sys
from bookmarkwidget import BookmarkWidget
from webengineview import WebEngineView
from historywindow import HistoryWindow
from PySide2 import QtCore
from PySide2.QtCore import QPoint, Qt, QUrl
from PySide2.QtWidgets import (QAction, QMenu, QTabBar, QTabWidget)
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem,
def _check_emit_enabled_changed(self, web_action, enabled):
    if enabled != self._actions_enabled[web_action]:
        self._actions_enabled[web_action] = enabled
        self.enabled_changed.emit(web_action, enabled)