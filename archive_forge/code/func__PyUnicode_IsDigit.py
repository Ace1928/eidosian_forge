from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _PyUnicode_IsDigit(ch):
    if _PyUnicode_ToDigit(ch) < 0:
        return 0
    return 1