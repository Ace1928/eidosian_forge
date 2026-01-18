from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@register_jitable
def _Py_ISUPPER(ch):
    """
    Equivalent to the CPython macro `Py_ISUPPER()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.UPPER