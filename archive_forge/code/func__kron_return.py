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
def _kron_return(a, b):
    a_is_arr = isinstance(a, types.Array)
    b_is_arr = isinstance(b, types.Array)
    if a_is_arr and b_is_arr:
        if a.ndim == 2 or b.ndim == 2:

            @register_jitable
            def ret(a, b, c):
                return c
            return ret
        else:

            @register_jitable
            def ret(a, b, c):
                return c.reshape(c.size)
            return ret
    elif a_is_arr:

        @register_jitable
        def ret(a, b, c):
            return c.reshape(a.shape)
        return ret
    elif b_is_arr:

        @register_jitable
        def ret(a, b, c):
            return c.reshape(b.shape)
        return ret
    else:

        @register_jitable
        def ret(a, b, c):
            return c[0]
        return ret