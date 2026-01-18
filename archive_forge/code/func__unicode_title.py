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
def _unicode_title(data, length, res, maxchars):
    """This is a translation of the function that titles a unicode string."""
    k = 0
    previous_cased = False
    mapped = np.empty(3, dtype=_Py_UCS4)
    for idx in range(length):
        mapped.fill(0)
        code_point = _get_code_point(data, idx)
        if previous_cased:
            n_res = _lower_ucs4(code_point, data, length, idx, mapped)
        else:
            n_res = _PyUnicode_ToTitleFull(_Py_UCS4(code_point), mapped)
        for m in mapped[:n_res]:
            maxchar, = maxchars
            maxchars[0] = max(maxchar, m)
            _set_code_point(res, k, m)
            k += 1
        previous_cased = _PyUnicode_IsCased(_Py_UCS4(code_point))
    return k