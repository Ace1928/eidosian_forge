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
def _atleast_nd_transform(min_ndim, axes):
    """
    Return a callback successively inserting 1-sized dimensions at the
    following axes.
    """
    assert min_ndim == len(axes)

    def transform(context, builder, arr, arrty, retty):
        for i in range(min_ndim):
            ndim = i + 1
            if arrty.ndim < ndim:
                axis = cgutils.intp_t(axes[i])
                newarrty = arrty.copy(ndim=arrty.ndim + 1)
                arr = expand_dims(context, builder, typing.signature(newarrty, arrty), (arr,), axis)
                arrty = newarrty
        return arr
    return transform