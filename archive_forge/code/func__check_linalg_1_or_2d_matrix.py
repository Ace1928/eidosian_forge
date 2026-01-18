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
def _check_linalg_1_or_2d_matrix(a, func_name, la_prefix=True):
    prefix = 'np.linalg' if la_prefix else 'np'
    interp = (prefix, func_name)
    if not isinstance(a, types.Array):
        raise TypingError('%s.%s() only supported for array types ' % interp)
    if not a.ndim <= 2:
        raise TypingError('%s.%s() only supported on 1 and 2-D arrays ' % interp)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        raise TypingError('%s.%s() only supported on float and complex arrays.' % interp)