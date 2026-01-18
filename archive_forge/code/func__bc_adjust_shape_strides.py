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
def _bc_adjust_shape_strides(context, builder, shapes, strides, target_shape):
    """
    Broadcast shapes and strides to target_shape given that their ndim already
    matches.  For each location where the shape is 1 and does not match the
    dim for target, it is set to the value at the target and the stride is
    set to zero.
    """
    bc_shapes = []
    bc_strides = []
    zero = context.get_constant(types.uintp, 0)
    one = context.get_constant(types.uintp, 1)
    mismatch = [builder.icmp_signed('!=', tar, old) for tar, old in zip(target_shape, shapes)]
    src_is_one = [builder.icmp_signed('==', old, one) for old in shapes]
    preds = [builder.and_(x, y) for x, y in zip(mismatch, src_is_one)]
    bc_shapes = [builder.select(p, tar, old) for p, tar, old in zip(preds, target_shape, shapes)]
    bc_strides = [builder.select(p, zero, old) for p, old in zip(preds, strides)]
    return (bc_shapes, bc_strides)