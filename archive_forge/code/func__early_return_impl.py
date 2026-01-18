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
@overload(_early_return)
def _early_return_impl(val):
    UNUSED = 0
    if isinstance(val, types.Complex):

        def impl(val):
            if np.isnan(val.real):
                if np.isnan(val.imag):
                    return (True, np.nan + np.nan * 1j)
                else:
                    return (True, np.nan + 0j)
            else:
                return (False, UNUSED)
    elif isinstance(val, types.Float):

        def impl(val):
            if np.isnan(val):
                return (True, np.nan)
            else:
                return (False, UNUSED)
    else:

        def impl(val):
            return (False, UNUSED)
    return impl