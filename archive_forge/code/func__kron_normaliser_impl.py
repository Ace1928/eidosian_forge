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
def _kron_normaliser_impl(x):
    if isinstance(x, types.Array):
        if x.layout not in ('C', 'F'):
            raise TypingError("np.linalg.kron only supports 'C' or 'F' layout input arrays. Received an input of layout '{}'.".format(x.layout))
        elif x.ndim == 2:

            @register_jitable
            def nrm_shape(x):
                xn = x.shape[-1]
                xm = x.shape[-2]
                return x.reshape(xm, xn)
            return nrm_shape
        else:

            @register_jitable
            def nrm_shape(x):
                xn = x.shape[-1]
                return x.reshape(1, xn)
            return nrm_shape
    else:

        @register_jitable
        def nrm_shape(x):
            a = np.empty((1, 1), type(x))
            a[0] = x
            return a
        return nrm_shape