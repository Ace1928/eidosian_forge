from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def _resize(self, increment):
    if self.size + increment > self.known_max:
        raise XcffibException('resizing memory buffer to be too big')
    self.size += increment