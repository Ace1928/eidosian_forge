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
def _get_slogdet_diag_walker(a):
    """
    Walks the diag of a LUP decomposed matrix
    uses that det(A) = prod(diag(lup(A)))
    and also that log(a)+log(b) = log(a*b)
    The return sign is adjusted based on the values found
    such that the log(value) stays in the real domain.
    """
    if isinstance(a.dtype, types.Complex):

        @register_jitable
        def cmplx_diag_walker(n, a, sgn):
            csgn = sgn + 0j
            acc = 0.0
            for k in range(n):
                absel = np.abs(a[k, k])
                csgn = csgn * (a[k, k] / absel)
                acc = acc + np.log(absel)
            return (csgn, acc)
        return cmplx_diag_walker
    else:

        @register_jitable
        def real_diag_walker(n, a, sgn):
            acc = 0.0
            for k in range(n):
                v = a[k, k]
                if v < 0.0:
                    sgn = -sgn
                    v = -v
                acc = acc + np.log(v)
            return (sgn + 0.0, acc)
        return real_diag_walker