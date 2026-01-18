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
def _bc_adjust_dimension(context, builder, shapes, strides, target_shape):
    """
    Preprocess dimension for broadcasting.
    Returns (shapes, strides) such that the ndim match *target_shape*.
    When expanding to higher ndim, the returning shapes and strides are
    prepended with ones and zeros, respectively.
    When truncating to lower ndim, the shapes are checked (in runtime).
    All extra dimension must have size of 1.
    """
    zero = context.get_constant(types.uintp, 0)
    one = context.get_constant(types.uintp, 1)
    if len(target_shape) > len(shapes):
        nd_diff = len(target_shape) - len(shapes)
        shapes = [one] * nd_diff + shapes
        strides = [zero] * nd_diff + strides
    elif len(target_shape) < len(shapes):
        nd_diff = len(shapes) - len(target_shape)
        dim_is_one = [builder.icmp_unsigned('==', sh, one) for sh in shapes[:nd_diff]]
        accepted = functools.reduce(builder.and_, dim_is_one, cgutils.true_bit)
        with builder.if_then(builder.not_(accepted), likely=False):
            msg = 'cannot broadcast source array for assignment'
            context.call_conv.return_user_exc(builder, ValueError, (msg,))
        shapes = shapes[nd_diff:]
        strides = strides[nd_diff:]
    return (shapes, strides)