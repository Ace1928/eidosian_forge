import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def _2d_tol_none_impl(A, tol=None):
    s = _compute_singular_values(A)
    r = A.shape[0]
    c = A.shape[1]
    l = max(r, c)
    t = s[0] * l * eps_val
    return _get_rank_from_singular_values(s, t)