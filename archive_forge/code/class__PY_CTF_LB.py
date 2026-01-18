from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
class _PY_CTF_LB(IntEnum):
    LINE_BREAK = 1
    LINE_FEED = 2
    CARRIAGE_RETURN = 4