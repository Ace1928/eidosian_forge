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
@register_jitable(_nrt=False)
def _set_code_point(a, i, ch):
    if a._kind == PY_UNICODE_1BYTE_KIND:
        set_uint8(a._data, i, ch)
    elif a._kind == PY_UNICODE_2BYTE_KIND:
        set_uint16(a._data, i, ch)
    elif a._kind == PY_UNICODE_4BYTE_KIND:
        set_uint32(a._data, i, ch)
    else:
        raise AssertionError('Unexpected unicode representation in _set_code_point')