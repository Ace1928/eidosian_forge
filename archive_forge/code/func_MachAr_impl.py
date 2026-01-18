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
@overload(np_MachAr)
def MachAr_impl():
    f = np_MachAr()
    _mach_ar_data = tuple([getattr(f, x) for x in _mach_ar_supported])
    if w:
        wmsg = w[0]
        warnings.warn_explicit(wmsg.message.args[0], NumbaDeprecationWarning, wmsg.filename, wmsg.lineno)

    def impl():
        return MachAr(*_mach_ar_data)
    return impl