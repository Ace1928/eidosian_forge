import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _sizef(self, prop):
    return QtCore.QSizeF(*float_list(prop))