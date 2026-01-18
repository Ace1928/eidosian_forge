from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def __DeviceTimeCoord_wrapper(typ, num_axes):

    def init(unpacker):
        i = typ(unpacker)
        i.num_axes = num_axes
        return i