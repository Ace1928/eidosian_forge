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
@overload(_oneD_norm_2)
def _oneD_norm_2_impl(a):
    nb_ret_type = getattr(a.dtype, 'underlying_float', a.dtype)
    np_ret_type = np_support.as_dtype(nb_ret_type)
    xxnrm2 = _BLAS().numba_xxnrm2(a.dtype)
    kind = ord(get_blas_kind(a.dtype, 'norm'))

    def impl(a):
        n = len(a)
        ret = np.empty((1,), dtype=np_ret_type)
        jmp = int(a.strides[0] / a.itemsize)
        r = xxnrm2(kind, n, a.ctypes, jmp, ret.ctypes)
        if r < 0:
            fatal_error_func()
            assert 0
        _dummy_liveness_func([ret.size, a.size])
        return ret[0]
    return impl