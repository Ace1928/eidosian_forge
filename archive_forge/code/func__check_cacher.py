from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
def _check_cacher(obj):
    if hasattr(obj, '_cacher'):
        parent = obj._cacher[1]()
        if parent is None:
            return False
        if hasattr(parent, '_item_cache'):
            if obj._cacher[0] in parent._item_cache:
                return obj is parent._item_cache[obj._cacher[0]]
    return False