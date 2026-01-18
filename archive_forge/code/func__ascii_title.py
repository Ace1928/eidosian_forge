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
def _ascii_title(data, res):
    """ Does .title() on an ASCII string """
    previous_is_cased = False
    for idx in range(len(data)):
        code_point = _get_code_point(data, idx)
        if _Py_ISLOWER(code_point):
            if not previous_is_cased:
                code_point = _Py_TOUPPER(code_point)
            previous_is_cased = True
        elif _Py_ISUPPER(code_point):
            if previous_is_cased:
                code_point = _Py_TOLOWER(code_point)
            previous_is_cased = True
        else:
            previous_is_cased = False
        _set_code_point(res, idx, code_point)