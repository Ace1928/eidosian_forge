import sys
from PySide2 import QtCore
from PySide2.QtCore import QDir, QFileInfo, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QDesktopServices
from PySide2.QtWidgets import (QAction, QLabel, QMenu, QProgressBar,
from PySide2.QtWebEngineWidgets import QWebEngineDownloadItem
def _update_tool_tip(self):
    path = self._download_item.path()
    tool_tip = '{}\n{}'.format(self._download_item.url().toString(), QDir.toNativeSeparators(path))
    total_bytes = self._download_item.total_bytes()
    if total_bytes > 0:
        tool_tip += '\n{}K'.format(total_bytes / 1024)
    state = self.state()
    if state == QWebEngineDownloadItem.DownloadRequested:
        tool_tip += '\n(requested)'
    elif state == QWebEngineDownloadItem.DownloadInProgress:
        tool_tip += '\n(downloading)'
    elif state == QWebEngineDownloadItem.DownloadCompleted:
        tool_tip += '\n(completed)'
    elif state == QWebEngineDownloadItem.DownloadCancelled:
        tool_tip += '\n(cancelled)'
    else:
        tool_tip += '\n(interrupted)'
    self.setToolTip(tool_tip)