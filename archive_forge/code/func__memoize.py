from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _memoize(f):

    class memodict(dict):

        def __missing__(self, key):
            ret = f(key)
            if isinstance(key, int) or len(key) == 1:
                self[key] = ret
            return ret
    return memodict().__getitem__