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
def _make_view(self, context, builder, indices, retty, arrty, arr, subiter):
    """
            Compute a 0d view for a given input array.
            """
    assert isinstance(retty, types.Array) and retty.ndim == 0
    ptr = subiter.compute_pointer(context, builder, indices, arrty, arr)
    view = context.make_array(retty)(context, builder)
    itemsize = get_itemsize(context, retty)
    shape = context.make_tuple(builder, types.UniTuple(types.intp, 0), ())
    strides = context.make_tuple(builder, types.UniTuple(types.intp, 0), ())
    populate_array(view, ptr, shape, strides, itemsize, meminfo=None)
    return view