import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def _from_str_dtype(dtype):
    m = re_typestr.match(dtype.str)
    if not m:
        raise NotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    if typecode == 'U':
        if dtype.byteorder not in '=|':
            raise NotImplementedError('Does not support non-native byteorder')
        count = dtype.itemsize // sizeof_unicode_char
        assert count == int(groups[1]), 'Unicode char size mismatch'
        return types.UnicodeCharSeq(count)
    elif typecode == 'S':
        count = dtype.itemsize
        assert count == int(groups[1]), 'Char size mismatch'
        return types.CharSeq(count)
    else:
        raise NotImplementedError(dtype)