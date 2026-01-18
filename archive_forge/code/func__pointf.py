import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _pointf(self, prop):
    return QtCore.QPointF(*float_list(prop))