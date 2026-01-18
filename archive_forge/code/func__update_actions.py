from functools import partial
import sys
from bookmarkwidget import BookmarkWidget
from webengineview import WebEngineView
from historywindow import HistoryWindow
from PySide2 import QtCore
from PySide2.QtCore import QPoint, Qt, QUrl
from PySide2.QtWidgets import (QAction, QMenu, QTabBar, QTabWidget)
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem,
def _update_actions(self, index):
    if index >= 0 and index < len(self._webengineviews):
        view = self._webengineviews[index]
        for web_action in WebEngineView.web_actions():
            enabled = view.is_web_action_enabled(web_action)
            self._check_emit_enabled_changed(web_action, enabled)