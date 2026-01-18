import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def _create_model(parent, serialized_bookmarks):
    result = QStandardItemModel(0, 1, parent)
    last_folder_item = None
    for entry in serialized_bookmarks:
        if len(entry) == 1:
            last_folder_item = _create_folder_item(entry[0])
            result.appendRow(last_folder_item)
        else:
            url = QUrl.fromUserInput(entry[0])
            title = entry[1]
            icon = QIcon(entry[2]) if len(entry) > 2 and entry[2] else None
            last_folder_item.appendRow(_create_item(url, title, icon))
    return result