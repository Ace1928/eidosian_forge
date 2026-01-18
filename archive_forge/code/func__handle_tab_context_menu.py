from functools import partial
import sys
from bookmarkwidget import BookmarkWidget
from webengineview import WebEngineView
from historywindow import HistoryWindow
from PySide2 import QtCore
from PySide2.QtCore import QPoint, Qt, QUrl
from PySide2.QtWidgets import (QAction, QMenu, QTabBar, QTabWidget)
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem,
def _handle_tab_context_menu(self, point):
    index = self.tabBar().tabAt(point)
    if index < 0:
        return
    tab_count = len(self._webengineviews)
    context_menu = QMenu()
    duplicate_tab_action = context_menu.addAction('Duplicate Tab')
    close_other_tabs_action = context_menu.addAction('Close Other Tabs')
    close_other_tabs_action.setEnabled(tab_count > 1)
    close_tabs_to_the_right_action = context_menu.addAction('Close Tabs to the Right')
    close_tabs_to_the_right_action.setEnabled(index < tab_count - 1)
    close_tab_action = context_menu.addAction('&Close Tab')
    chosen_action = context_menu.exec_(self.tabBar().mapToGlobal(point))
    if chosen_action == duplicate_tab_action:
        current_url = self.url()
        self.add_browser_tab().load(current_url)
    elif chosen_action == close_other_tabs_action:
        for t in range(tab_count - 1, -1, -1):
            if t != index:
                self.handle_tab_close_request(t)
    elif chosen_action == close_tabs_to_the_right_action:
        for t in range(tab_count - 1, index, -1):
            self.handle_tab_close_request(t)
    elif chosen_action == close_tab_action:
        self.handle_tab_close_request(index)