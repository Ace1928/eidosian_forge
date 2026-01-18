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
def _searchsorted(func_1, func_2):

    def impl(a, v):
        min_idx = 0
        max_idx = len(a)
        out = np.empty(v.size, np.intp)
        last_key_val = v.flat[0]
        for i in range(v.size):
            key_val = v.flat[i]
            if func_1(last_key_val, key_val):
                max_idx = len(a)
            else:
                min_idx = 0
                if max_idx < len(a):
                    max_idx += 1
                else:
                    max_idx = len(a)
            last_key_val = key_val
            while min_idx < max_idx:
                mid_idx = min_idx + (max_idx - min_idx >> 1)
                mid_val = a[mid_idx]
                if func_2(mid_val, key_val):
                    min_idx = mid_idx + 1
                else:
                    max_idx = mid_idx
            out[i] = min_idx
        return out.reshape(v.shape)
    return impl