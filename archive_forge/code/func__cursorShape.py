import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _cursorShape(self, prop):
    return QtGui.QCursor(getattr(QtCore.Qt, prop.text))