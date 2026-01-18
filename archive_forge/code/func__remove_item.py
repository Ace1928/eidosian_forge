import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def _remove_item(self, item):
    button = QMessageBox.question(self, 'Remove', 'Would you like to remove "{}"?'.format(item.text()), QMessageBox.Yes | QMessageBox.No)
    if button == QMessageBox.Yes:
        item.parent().removeRow(item.row())