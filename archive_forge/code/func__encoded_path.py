import locale
import sys
from PyQt5.QtCore import (PYQT_VERSION_STR, QDir, QFile, QFileInfo, QIODevice,
from .pylupdate import *
def _encoded_path(path):
    return path.encode(locale.getdefaultlocale()[1])