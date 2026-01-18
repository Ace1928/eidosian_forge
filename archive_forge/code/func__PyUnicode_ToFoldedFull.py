from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _PyUnicode_ToFoldedFull(ch, res):
    ctype = _PyUnicode_gettyperecord(ch)
    extended_case_mask = _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK
    if ctype.flags & extended_case_mask and ctype.lower >> 20 & 7:
        index = (ctype.lower & 65535) + (ctype.lower >> 24)
        n = ctype.lower >> 20 & 7
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    return _PyUnicode_ToLowerFull(ch, res)