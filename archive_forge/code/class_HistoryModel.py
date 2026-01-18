from PySide2.QtWebEngineWidgets import (QWebEnginePage, QWebEngineView,
from PySide2.QtWidgets import QApplication, QDesktopWidget, QTreeView
from PySide2.QtCore import (Signal, QAbstractTableModel, QModelIndex, Qt,
class HistoryModel(QAbstractTableModel):

    def __init__(self, history, parent=None):
        super(HistoryModel, self).__init__(parent)
        self._history = history

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return 'Title' if section == 0 else 'Url'
        return None

    def rowCount(self, index=QModelIndex()):
        return self._history.count()

    def columnCount(self, index=QModelIndex()):
        return 2

    def item_at(self, model_index):
        return self._history.itemAt(model_index.row())

    def data(self, index, role=Qt.DisplayRole):
        item = self.item_at(index)
        column = index.column()
        if role == Qt.DisplayRole:
            return item.title() if column == 0 else item.url().toString()
        return None

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()