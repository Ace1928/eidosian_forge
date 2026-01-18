import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _point(self, prop):
    return QtCore.QPoint(*int_list(prop))