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
def _create_menu(self):
    file_menu = self.menuBar().addMenu('&File')
    exit_action = QAction(QIcon.fromTheme('application-exit'), 'E&xit', self, shortcut='Ctrl+Q', triggered=qApp.quit)
    file_menu.addAction(exit_action)
    navigation_menu = self.menuBar().addMenu('&Navigation')
    style_icons = ':/qt-project.org/styles/commonstyle/images/'
    back_action = QAction(QIcon.fromTheme('go-previous', QIcon(style_icons + 'left-32.png')), 'Back', self, shortcut=QKeySequence(QKeySequence.Back), triggered=self._tab_widget.back)
    self._actions[QWebEnginePage.Back] = back_action
    back_action.setEnabled(False)
    navigation_menu.addAction(back_action)
    forward_action = QAction(QIcon.fromTheme('go-next', QIcon(style_icons + 'right-32.png')), 'Forward', self, shortcut=QKeySequence(QKeySequence.Forward), triggered=self._tab_widget.forward)
    forward_action.setEnabled(False)
    self._actions[QWebEnginePage.Forward] = forward_action
    navigation_menu.addAction(forward_action)
    reload_action = QAction(QIcon(style_icons + 'refresh-32.png'), 'Reload', self, shortcut=QKeySequence(QKeySequence.Refresh), triggered=self._tab_widget.reload)
    self._actions[QWebEnginePage.Reload] = reload_action
    reload_action.setEnabled(False)
    navigation_menu.addAction(reload_action)
    navigation_menu.addSeparator()
    new_tab_action = QAction('New Tab', self, shortcut='Ctrl+T', triggered=self.add_browser_tab)
    navigation_menu.addAction(new_tab_action)
    close_tab_action = QAction('Close Current Tab', self, shortcut='Ctrl+W', triggered=self._close_current_tab)
    navigation_menu.addAction(close_tab_action)
    navigation_menu.addSeparator()
    history_action = QAction('History...', self, triggered=self._tab_widget.show_history)
    navigation_menu.addAction(history_action)
    edit_menu = self.menuBar().addMenu('&Edit')
    find_action = QAction('Find', self, shortcut=QKeySequence(QKeySequence.Find), triggered=self._show_find)
    edit_menu.addAction(find_action)
    edit_menu.addSeparator()
    undo_action = QAction('Undo', self, shortcut=QKeySequence(QKeySequence.Undo), triggered=self._tab_widget.undo)
    self._actions[QWebEnginePage.Undo] = undo_action
    undo_action.setEnabled(False)
    edit_menu.addAction(undo_action)
    redo_action = QAction('Redo', self, shortcut=QKeySequence(QKeySequence.Redo), triggered=self._tab_widget.redo)
    self._actions[QWebEnginePage.Redo] = redo_action
    redo_action.setEnabled(False)
    edit_menu.addAction(redo_action)
    edit_menu.addSeparator()
    cut_action = QAction('Cut', self, shortcut=QKeySequence(QKeySequence.Cut), triggered=self._tab_widget.cut)
    self._actions[QWebEnginePage.Cut] = cut_action
    cut_action.setEnabled(False)
    edit_menu.addAction(cut_action)
    copy_action = QAction('Copy', self, shortcut=QKeySequence(QKeySequence.Copy), triggered=self._tab_widget.copy)
    self._actions[QWebEnginePage.Copy] = copy_action
    copy_action.setEnabled(False)
    edit_menu.addAction(copy_action)
    paste_action = QAction('Paste', self, shortcut=QKeySequence(QKeySequence.Paste), triggered=self._tab_widget.paste)
    self._actions[QWebEnginePage.Paste] = paste_action
    paste_action.setEnabled(False)
    edit_menu.addAction(paste_action)
    edit_menu.addSeparator()
    select_all_action = QAction('Select All', self, shortcut=QKeySequence(QKeySequence.SelectAll), triggered=self._tab_widget.select_all)
    self._actions[QWebEnginePage.SelectAll] = select_all_action
    select_all_action.setEnabled(False)
    edit_menu.addAction(select_all_action)
    self._bookmark_menu = self.menuBar().addMenu('&Bookmarks')
    add_bookmark_action = QAction('&Add Bookmark', self, triggered=self._add_bookmark)
    self._bookmark_menu.addAction(add_bookmark_action)
    add_tool_bar_bookmark_action = QAction('&Add Bookmark to Tool Bar', self, triggered=self._add_tool_bar_bookmark)
    self._bookmark_menu.addAction(add_tool_bar_bookmark_action)
    self._bookmark_menu.addSeparator()
    tools_menu = self.menuBar().addMenu('&Tools')
    download_action = QAction('Open Downloads', self, triggered=DownloadWidget.open_download_directory)
    tools_menu.addAction(download_action)
    window_menu = self.menuBar().addMenu('&Window')
    window_menu.addAction(self._bookmark_dock.toggleViewAction())
    window_menu.addSeparator()
    zoom_in_action = QAction(QIcon.fromTheme('zoom-in'), 'Zoom In', self, shortcut=QKeySequence(QKeySequence.ZoomIn), triggered=self._zoom_in)
    window_menu.addAction(zoom_in_action)
    zoom_out_action = QAction(QIcon.fromTheme('zoom-out'), 'Zoom Out', self, shortcut=QKeySequence(QKeySequence.ZoomOut), triggered=self._zoom_out)
    window_menu.addAction(zoom_out_action)
    reset_zoom_action = QAction(QIcon.fromTheme('zoom-original'), 'Reset Zoom', self, shortcut='Ctrl+0', triggered=self._reset_zoom)
    window_menu.addAction(reset_zoom_action)
    about_menu = self.menuBar().addMenu('&About')
    about_action = QAction('About Qt', self, shortcut=QKeySequence(QKeySequence.HelpContents), triggered=qApp.aboutQt)
    about_menu.addAction(about_action)