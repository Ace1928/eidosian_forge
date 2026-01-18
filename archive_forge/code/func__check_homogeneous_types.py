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
def _check_homogeneous_types(func_name, *types):
    t0 = types[0].dtype
    for t in types[1:]:
        if t.dtype != t0:
            msg = 'np.linalg.%s() only supports inputs that have homogeneous dtypes.' % func_name
            raise TypingError(msg, highlighting=False)