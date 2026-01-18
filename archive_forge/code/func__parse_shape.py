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
def _parse_shape(context, builder, ty, val):
    """
    Parse the shape argument to an array constructor.
    """

    def safecast_intp(context, builder, src_t, src):
        """Cast src to intp only if value can be maintained"""
        intp_t = context.get_value_type(types.intp)
        intp_width = intp_t.width
        intp_ir = ir.IntType(intp_width)
        maxval = Constant(intp_ir, (1 << intp_width - 1) - 1)
        if src_t.width < intp_width:
            res = builder.sext(src, intp_ir)
        elif src_t.width >= intp_width:
            is_larger = builder.icmp_signed('>', src, maxval)
            with builder.if_then(is_larger, likely=False):
                context.call_conv.return_user_exc(builder, ValueError, ('Cannot safely convert value to intp',))
            if src_t.width > intp_width:
                res = builder.trunc(src, intp_ir)
            else:
                res = src
        return res
    if isinstance(ty, types.Integer):
        ndim = 1
        passed_shapes = [context.cast(builder, val, ty, types.intp)]
    else:
        assert isinstance(ty, types.BaseTuple)
        ndim = ty.count
        passed_shapes = cgutils.unpack_tuple(builder, val, count=ndim)
    shapes = []
    for s in passed_shapes:
        shapes.append(safecast_intp(context, builder, s.type, s))
    zero = context.get_constant_generic(builder, types.intp, 0)
    for dim in range(ndim):
        is_neg = builder.icmp_signed('<', shapes[dim], zero)
        with cgutils.if_unlikely(builder, is_neg):
            context.call_conv.return_user_exc(builder, ValueError, ('negative dimensions not allowed',))
    return shapes