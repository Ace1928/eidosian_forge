import json, os, warnings
from PySide2 import QtCore
from PySide2.QtCore import (QDir, QFileInfo, QModelIndex, QStandardPaths, Qt,
from PySide2.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide2.QtWidgets import (QAction, QDockWidget, QMenu, QMessageBox,
def _read_bookmarks(self):
    bookmark_file_name = os.path.join(QDir.toNativeSeparators(_config_dir()), _bookmark_file)
    if os.path.exists(bookmark_file_name):
        print('Reading {}...'.format(bookmark_file_name))
        return json.load(open(bookmark_file_name))
    return _default_bookmarks