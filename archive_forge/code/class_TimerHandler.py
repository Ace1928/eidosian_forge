import sys
from PySide2.QtCore import (Qt, QByteArray, QModelIndex, QObject, QTimer, QUrl)
from PySide2.QtGui import (QColor, QStandardItemModel, QStandardItem)
from PySide2.QtWidgets import (QApplication, QTreeView)
from PySide2.QtRemoteObjects import (QRemoteObjectHost, QRemoteObjectNode,
class TimerHandler(QObject):

    def __init__(self, model):
        super(TimerHandler, self).__init__()
        self._model = model

    def change_data(self):
        for i in range(10, 50):
            self._model.setData(self._model.index(i, 1), QColor(Qt.blue), Qt.BackgroundRole)

    def insert_data(self):
        self._model.insertRows(2, 9)
        for i in range(2, 11):
            self._model.setData(self._model.index(i, 1), QColor(Qt.green), Qt.BackgroundRole)
            self._model.setData(self._model.index(i, 1), 'InsertedRow', Qt.DisplayRole)

    def remove_data(self):
        self._model.removeRows(2, 4)

    def change_flags(self):
        item = self._model.item(0, 0)
        item.setEnabled(False)
        item = item.child(0, 0)
        item.setFlags(item.flags() & Qt.ItemIsSelectable)

    def move_data(self):
        self._model.moveRows(QModelIndex(), 2, 4, QModelIndex(), 10)