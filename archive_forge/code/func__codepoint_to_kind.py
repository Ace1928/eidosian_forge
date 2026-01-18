import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@register_jitable
def _codepoint_to_kind(cp):
    """
    Compute the minimum unicode kind needed to hold a given codepoint
    """
    if cp < 256:
        return PY_UNICODE_1BYTE_KIND
    elif cp < 65536:
        return PY_UNICODE_2BYTE_KIND
    else:
        MAX_UNICODE = 1114111
        if cp > MAX_UNICODE:
            msg = 'Invalid codepoint. Found value greater than Unicode maximum'
            raise ValueError(msg)
        return PY_UNICODE_4BYTE_KIND