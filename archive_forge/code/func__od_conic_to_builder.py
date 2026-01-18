import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _od_conic_to_builder(self, cb):
    if cb is None:
        return self._od_conic_to_noop

    def conic_to(a, b, c):
        return cb(a[0], b[0], c) or 0
    return FT_Outline_ConicToFunc(conic_to)