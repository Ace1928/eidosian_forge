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
def _dtype_of_compound(inobj):
    obj = inobj
    while True:
        if isinstance(obj, (types.Number, types.Boolean)):
            return as_dtype(obj)
        l = getattr(obj, '__len__', None)
        if l is not None and l() == 0:
            return np.float64
        dt = getattr(obj, 'dtype', None)
        if dt is None:
            raise TypeError('type has no dtype attr')
        if isinstance(obj, types.Sequence):
            obj = obj.dtype
        else:
            return as_dtype(dt)