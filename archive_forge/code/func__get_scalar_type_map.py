import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def _get_scalar_type_map():
    """
        Return a dictionary mapping native endian scalar dtype to ctypes types
        """
    ct = ctypes
    simple_types = [ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong, ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong, ct.c_float, ct.c_double, ct.c_bool]
    return {_dtype(ctype): ctype for ctype in simple_types}