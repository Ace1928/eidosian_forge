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
def _pick_kind(kind1, kind2):
    if kind1 == PY_UNICODE_WCHAR_KIND or kind2 == PY_UNICODE_WCHAR_KIND:
        raise AssertionError('PY_UNICODE_WCHAR_KIND unsupported')
    if kind1 == PY_UNICODE_1BYTE_KIND:
        return kind2
    elif kind1 == PY_UNICODE_2BYTE_KIND:
        if kind2 == PY_UNICODE_4BYTE_KIND:
            return kind2
        else:
            return kind1
    elif kind1 == PY_UNICODE_4BYTE_KIND:
        return kind1
    else:
        raise AssertionError('Unexpected unicode representation in _pick_kind')