from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def _add_ext(key, value, events, errors):
    if not issubclass(value, Extension):
        raise XcffibException('Extension type not derived from xcffib.Extension')
    extensions[key] = (value, events, errors)