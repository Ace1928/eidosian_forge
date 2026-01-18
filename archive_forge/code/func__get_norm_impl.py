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
def _get_norm_impl(x, ord_flag):
    nb_ret_type = getattr(x.dtype, 'underlying_float', x.dtype)
    np_ret_type = np_support.as_dtype(nb_ret_type)
    np_dtype = np_support.as_dtype(x.dtype)
    xxnrm2 = _BLAS().numba_xxnrm2(x.dtype)
    kind = ord(get_blas_kind(x.dtype, 'norm'))
    if x.ndim == 1:
        if ord_flag in (None, types.none):

            def oneD_impl(x, ord=None):
                return _oneD_norm_2(x)
        else:

            def oneD_impl(x, ord=None):
                n = len(x)
                if n == 0:
                    return 0.0
                if ord == 2:
                    return _oneD_norm_2(x)
                elif ord == np.inf:
                    ret = abs(x[0])
                    for k in range(1, n):
                        val = abs(x[k])
                        if val > ret:
                            ret = val
                    return ret
                elif ord == -np.inf:
                    ret = abs(x[0])
                    for k in range(1, n):
                        val = abs(x[k])
                        if val < ret:
                            ret = val
                    return ret
                elif ord == 0:
                    ret = 0.0
                    for k in range(n):
                        if x[k] != 0.0:
                            ret += 1.0
                    return ret
                elif ord == 1:
                    ret = 0.0
                    for k in range(n):
                        ret += abs(x[k])
                    return ret
                else:
                    ret = 0.0
                    for k in range(n):
                        ret += abs(x[k]) ** ord
                    return ret ** (1.0 / ord)
        return oneD_impl
    elif x.ndim == 2:
        if ord_flag in (None, types.none):
            if x.layout == 'C':

                @register_jitable
                def array_prepare(x):
                    return x
            elif x.layout == 'F':

                @register_jitable
                def array_prepare(x):
                    return x.T
            else:

                @register_jitable
                def array_prepare(x):
                    return x.copy()

            def twoD_impl(x, ord=None):
                n = x.size
                if n == 0:
                    return 0.0
                x_c = array_prepare(x)
                return _oneD_norm_2(x_c.reshape(n))
        else:
            max_val = np.finfo(np_ret_type.type).max

            def twoD_impl(x, ord=None):
                n = x.shape[-1]
                m = x.shape[-2]
                if x.size == 0:
                    return 0.0
                if ord == np.inf:
                    global_max = 0.0
                    for ii in range(m):
                        tmp = 0.0
                        for jj in range(n):
                            tmp += abs(x[ii, jj])
                        if tmp > global_max:
                            global_max = tmp
                    return global_max
                elif ord == -np.inf:
                    global_min = max_val
                    for ii in range(m):
                        tmp = 0.0
                        for jj in range(n):
                            tmp += abs(x[ii, jj])
                        if tmp < global_min:
                            global_min = tmp
                    return global_min
                elif ord == 1:
                    global_max = 0.0
                    for ii in range(n):
                        tmp = 0.0
                        for jj in range(m):
                            tmp += abs(x[jj, ii])
                        if tmp > global_max:
                            global_max = tmp
                    return global_max
                elif ord == -1:
                    global_min = max_val
                    for ii in range(n):
                        tmp = 0.0
                        for jj in range(m):
                            tmp += abs(x[jj, ii])
                        if tmp < global_min:
                            global_min = tmp
                    return global_min
                elif ord == 2:
                    return _compute_singular_values(x)[0]
                elif ord == -2:
                    return _compute_singular_values(x)[-1]
                else:
                    raise ValueError('Invalid norm order for matrices.')
        return twoD_impl
    else:
        assert 0