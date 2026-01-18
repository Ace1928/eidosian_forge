from PySide2.QtWebEngineWidgets import (QWebEnginePage, QWebEngineView,
from PySide2.QtWidgets import QApplication, QDesktopWidget, QTreeView
from PySide2.QtCore import (Signal, QAbstractTableModel, QModelIndex, Qt,
class HistoryWindow(QTreeView):
    open_url = Signal(QUrl)

    def __init__(self, history, parent):
        super(HistoryWindow, self).__init__(parent)
        self._model = HistoryModel(history, self)
        self.setModel(self._model)
        self.activated.connect(self._activated)
        screen = QApplication.desktop().screenGeometry(parent)
        self.resize(screen.width() / 3, screen.height() / 3)
        self._adjustSize()

    def refresh(self):
        self._model.refresh()
        self._adjustSize()

    def _adjustSize(self):
        if self._model.rowCount() > 0:
            self.resizeColumnToContents(0)

    def _activated(self, index):
        item = self._model.item_at(index)
        self.open_url.emit(item.url())