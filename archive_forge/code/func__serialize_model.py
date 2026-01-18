import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def _serialize_model(model, directory):
    result = []
    folder_count = model.rowCount()
    for f in range(0, folder_count):
        folder_item = model.item(f)
        result.append([folder_item.text()])
        item_count = folder_item.rowCount()
        for i in range(0, item_count):
            item = folder_item.child(i)
            entry = [item.data(_url_role).toString(), item.text()]
            icon = item.icon()
            if not icon.isNull():
                icon_sizes = icon.availableSizes()
                largest_size = icon_sizes[len(icon_sizes) - 1]
                icon_file_name = '{}/icon{:02}_{:02}_{}.png'.format(directory, f, i, largest_size.width())
                icon.pixmap(largest_size).save(icon_file_name, 'PNG')
                entry.append(icon_file_name)
            result.append(entry)
    return result