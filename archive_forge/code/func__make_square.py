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
@register_jitable
def _make_square(m):
    """
    Takes a 1d array and tiles it to form a square matrix
    - i.e. a facsimile of np.tile(m, (len(m), 1))
    """
    assert m.ndim == 1
    len_m = len(m)
    out = np.empty((len_m, len_m), dtype=m.dtype)
    for i in range(len_m):
        out[i] = m
    return out