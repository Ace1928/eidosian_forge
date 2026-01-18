import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def _ctype_from_dtype_scalar(dtype):
    dtype_with_endian = dtype.newbyteorder('S').newbyteorder('S')
    dtype_native = dtype.newbyteorder('=')
    try:
        ctype = _scalar_type_map[dtype_native]
    except KeyError as e:
        raise NotImplementedError('Converting {!r} to a ctypes type'.format(dtype)) from None
    if dtype_with_endian.byteorder == '>':
        ctype = ctype.__ctype_be__
    elif dtype_with_endian.byteorder == '<':
        ctype = ctype.__ctype_le__
    return ctype