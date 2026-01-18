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
def _prepare_cov_input_inner(m, y, rowvar, dtype):
    m_arr = np.atleast_2d(_asarray(m))
    y_arr = np.atleast_2d(_asarray(y))
    if not rowvar:
        if m_arr.shape[0] != 1:
            m_arr = m_arr.T
        if y_arr.shape[0] != 1:
            y_arr = y_arr.T
    m_rows, m_cols = m_arr.shape
    y_rows, y_cols = y_arr.shape
    if m_cols != y_cols:
        raise ValueError('m and y have incompatible dimensions')
    out = np.empty((m_rows + y_rows, m_cols), dtype=dtype)
    out[:m_rows, :] = m_arr
    out[-y_rows:, :] = y_arr
    return out