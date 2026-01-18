from PySide2.QtWebEngineWidgets import (QWebEnginePage, QWebEngineView,
from PySide2.QtWidgets import QApplication, QDesktopWidget, QTreeView
from PySide2.QtCore import (Signal, QAbstractTableModel, QModelIndex, Qt,
def _adjustSize(self):
    if self._model.rowCount() > 0:
        self.resizeColumnToContents(0)