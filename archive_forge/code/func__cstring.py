import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _cstring(self, prop):
    return str(prop.text)