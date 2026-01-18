import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def _arange_dtype(*args):
    bounds = [a for a in args if not isinstance(a, types.NoneType)]
    if any((isinstance(a, types.Complex) for a in bounds)):
        dtype = types.complex128
    elif any((isinstance(a, types.Float) for a in bounds)):
        dtype = types.float64
    else:
        NPY_TY = getattr(types, 'int%s' % (8 * np.dtype(int).itemsize))
        unliteral_bounds = [types.unliteral(x) for x in bounds]
        dtype = max(unliteral_bounds + [NPY_TY])
    return dtype