from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
class _PyUnicode_TyperecordMasks(IntEnum):
    ALPHA_MASK = 1
    DECIMAL_MASK = 2
    DIGIT_MASK = 4
    LOWER_MASK = 8
    LINEBREAK_MASK = 16
    SPACE_MASK = 32
    TITLE_MASK = 64
    UPPER_MASK = 128
    XID_START_MASK = 256
    XID_CONTINUE_MASK = 512
    PRINTABLE_MASK = 1024
    NUMERIC_MASK = 2048
    CASE_IGNORABLE_MASK = 4096
    CASED_MASK = 8192
    EXTENDED_CASE_MASK = 16384