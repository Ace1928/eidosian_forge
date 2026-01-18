from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _Py_ISLINEBREAK(ch):
    """Check if character is ASCII line break"""
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.LINE_BREAK