import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def _make_write_func(file_obj):
    """Return a CFFI callback that writes to a file-like object."""
    if file_obj is None:
        return ffi.NULL

    @ffi.callback('cairo_write_func_t', error=constants.STATUS_WRITE_ERROR)
    def write_func(_closure, data, length):
        file_obj.write(ffi.buffer(data, length))
        return constants.STATUS_SUCCESS
    return write_func