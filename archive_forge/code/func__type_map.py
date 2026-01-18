from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
def _type_map():
    """
    Lazily compute type map, as calling ffi.typeof() involves costly
    parsing of C code...
    """
    global _cached_type_map
    if _cached_type_map is None:
        _cached_type_map = {ffi.typeof('bool'): types.boolean, ffi.typeof('char'): types.char, ffi.typeof('short'): types.short, ffi.typeof('int'): types.intc, ffi.typeof('long'): types.long_, ffi.typeof('long long'): types.longlong, ffi.typeof('unsigned char'): types.uchar, ffi.typeof('unsigned short'): types.ushort, ffi.typeof('unsigned int'): types.uintc, ffi.typeof('unsigned long'): types.ulong, ffi.typeof('unsigned long long'): types.ulonglong, ffi.typeof('int8_t'): types.char, ffi.typeof('uint8_t'): types.uchar, ffi.typeof('int16_t'): types.short, ffi.typeof('uint16_t'): types.ushort, ffi.typeof('int32_t'): types.intc, ffi.typeof('uint32_t'): types.uintc, ffi.typeof('int64_t'): types.longlong, ffi.typeof('uint64_t'): types.ulonglong, ffi.typeof('float'): types.float_, ffi.typeof('double'): types.double, ffi.typeof('ssize_t'): types.intp, ffi.typeof('size_t'): types.uintp, ffi.typeof('void'): types.void}
    return _cached_type_map