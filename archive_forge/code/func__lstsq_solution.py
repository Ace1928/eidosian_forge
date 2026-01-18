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
def _lstsq_solution(b, bcpy, n):
    """
    Extract 'x' (the lstsq solution) from the 'bcpy' scratch space.
    Note 'b' is only used to check the system input dimension...
    """
    raise NotImplementedError