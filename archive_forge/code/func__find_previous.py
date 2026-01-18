from PySide2 import QtCore
from PySide2.QtCore import Qt, QUrl
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QCheckBox, QDockWidget, QHBoxLayout,
from PySide2.QtWebEngineWidgets import QWebEnginePage
def _find_previous(self):
    self._emit_find(True)