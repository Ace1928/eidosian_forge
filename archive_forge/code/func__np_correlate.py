import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(np.correlate)
def _np_correlate(a, v, mode='valid'):
    _assert_1d(a, 'np.correlate')
    _assert_1d(v, 'np.correlate')

    @register_jitable
    def op_conj(x):
        return np.conj(x)

    @register_jitable
    def op_nop(x):
        return x
    if a.dtype in types.complex_domain:
        if v.dtype in types.complex_domain:
            a_op = op_nop
            b_op = op_conj
        else:
            a_op = op_nop
            b_op = op_nop
    elif v.dtype in types.complex_domain:
        a_op = op_nop
        b_op = op_conj
    else:
        a_op = op_conj
        b_op = op_nop

    def impl(a, v, mode='valid'):
        la = len(a)
        lv = len(v)
        if la == 0:
            raise ValueError("'a' cannot be empty")
        if lv == 0:
            raise ValueError("'v' cannot be empty")
        if la < lv:
            return _np_correlate_core(b_op(v), a_op(a), mode, -1)
        else:
            return _np_correlate_core(a_op(a), b_op(v), mode, 1)
    return impl