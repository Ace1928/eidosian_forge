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
def _np_stack(context, builder, arrtys, arrs, retty, axis):
    ndim = retty.ndim
    zero = cgutils.intp_t(0)
    one = cgutils.intp_t(1)
    ll_narrays = cgutils.intp_t(len(arrs))
    arrs = [make_array(aty)(context, builder, value=a) for aty, a in zip(arrtys, arrs)]
    axis = _normalize_axis(context, builder, 'np.stack', ndim, axis)
    orig_shape = cgutils.unpack_tuple(builder, arrs[0].shape)
    for arr in arrs[1:]:
        is_ok = cgutils.true_bit
        for sh, orig_sh in zip(cgutils.unpack_tuple(builder, arr.shape), orig_shape):
            is_ok = builder.and_(is_ok, builder.icmp_signed('==', sh, orig_sh))
            with builder.if_then(builder.not_(is_ok), likely=False):
                context.call_conv.return_user_exc(builder, ValueError, ('np.stack(): all input arrays must have the same shape',))
    orig_strides = [cgutils.unpack_tuple(builder, arr.strides) for arr in arrs]
    ll_shty = ir.ArrayType(cgutils.intp_t, ndim)
    input_shapes = cgutils.alloca_once(builder, ll_shty)
    ret_shapes = cgutils.alloca_once(builder, ll_shty)
    for dim in range(ndim - 1):
        ll_dim = cgutils.intp_t(dim)
        after_axis = builder.icmp_signed('>=', ll_dim, axis)
        sh = orig_shape[dim]
        idx = builder.select(after_axis, builder.add(ll_dim, one), ll_dim)
        builder.store(sh, cgutils.gep_inbounds(builder, input_shapes, 0, idx))
        builder.store(sh, cgutils.gep_inbounds(builder, ret_shapes, 0, idx))
    builder.store(one, cgutils.gep_inbounds(builder, input_shapes, 0, axis))
    builder.store(ll_narrays, cgutils.gep_inbounds(builder, ret_shapes, 0, axis))
    input_shapes = cgutils.unpack_tuple(builder, builder.load(input_shapes))
    input_shapes = [input_shapes] * len(arrs)
    ret_shapes = cgutils.unpack_tuple(builder, builder.load(ret_shapes))
    input_strides = [cgutils.alloca_once(builder, ll_shty) for i in range(len(arrs))]
    for dim in range(ndim - 1):
        ll_dim = cgutils.intp_t(dim)
        after_axis = builder.icmp_signed('>=', ll_dim, axis)
        idx = builder.select(after_axis, builder.add(ll_dim, one), ll_dim)
        for i in range(len(arrs)):
            builder.store(orig_strides[i][dim], cgutils.gep_inbounds(builder, input_strides[i], 0, idx))
    for i in range(len(arrs)):
        builder.store(zero, cgutils.gep_inbounds(builder, input_strides[i], 0, axis))
    input_strides = [cgutils.unpack_tuple(builder, builder.load(st)) for st in input_strides]
    ret = _do_concatenate(context, builder, axis, arrtys, arrs, input_shapes, input_strides, retty, ret_shapes)
    return impl_ret_new_ref(context, builder, retty, ret._getvalue())