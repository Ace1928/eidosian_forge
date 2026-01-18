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
def _get_matrix_rank_impl(A, tol):
    ndim = A.ndim
    if ndim == 1:

        def _1d_matrix_rank_impl(A, tol=None):
            for k in range(len(A)):
                if A[k] != 0.0:
                    return 1
            return 0
        return _1d_matrix_rank_impl
    elif ndim == 2:
        return _2d_matrix_rank_impl(A, tol)
    else:
        assert 0