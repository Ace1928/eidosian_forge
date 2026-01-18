from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
class _PY_CTF(IntEnum):
    LOWER = 1
    UPPER = 2
    ALPHA = 1 | 2
    DIGIT = 4
    ALNUM = 1 | 2 | 4
    SPACE = 8
    XDIGIT = 16