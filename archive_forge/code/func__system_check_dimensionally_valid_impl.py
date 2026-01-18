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
@overload(_system_check_dimensionally_valid)
def _system_check_dimensionally_valid_impl(a, b):
    ndim = b.ndim
    if ndim == 1:

        def oneD_impl(a, b):
            am = a.shape[-2]
            bm = b.shape[-1]
            if am != bm:
                raise np.linalg.LinAlgError('Incompatible array sizes, system is not dimensionally valid.')
        return oneD_impl
    else:

        def twoD_impl(a, b):
            am = a.shape[-2]
            bm = b.shape[-2]
            if am != bm:
                raise np.linalg.LinAlgError('Incompatible array sizes, system is not dimensionally valid.')
        return twoD_impl