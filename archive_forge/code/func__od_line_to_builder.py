import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _od_line_to_builder(self, cb):
    if cb is None:
        return self._od_line_to_noop

    def line_to(a, b):
        return cb(a[0], b) or 0
    return FT_Outline_LineToFunc(line_to)