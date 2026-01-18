from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def _process_error(self, c_error):
    self.invalid()
    if c_error != ffi.NULL:
        error = self._error_offsets[c_error.error_code]
        buf = CffiUnpacker(c_error)
        raise error(buf)