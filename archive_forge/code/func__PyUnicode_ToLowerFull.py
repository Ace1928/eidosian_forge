from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _PyUnicode_ToLowerFull(ch, res):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.lower & 65535
        n = ctype.lower >> 24
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.lower
    return 1