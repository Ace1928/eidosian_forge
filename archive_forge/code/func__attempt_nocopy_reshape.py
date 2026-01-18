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
def _attempt_nocopy_reshape(context, builder, aryty, ary, newnd, newshape, newstrides):
    """
    Call into Numba_attempt_nocopy_reshape() for the given array type
    and instance, and the specified new shape.

    Return value is non-zero if successful, and the array pointed to
    by *newstrides* will be filled up with the computed results.
    """
    ll_intp = context.get_value_type(types.intp)
    ll_intp_star = ll_intp.as_pointer()
    ll_intc = context.get_value_type(types.intc)
    fnty = ir.FunctionType(ll_intc, [ll_intp, ll_intp_star, ll_intp_star, ll_intp, ll_intp_star, ll_intp_star, ll_intp, ll_intc])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_attempt_nocopy_reshape')
    nd = ll_intp(aryty.ndim)
    shape = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('shape'), 0, 0)
    strides = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('strides'), 0, 0)
    newnd = ll_intp(newnd)
    newshape = cgutils.gep_inbounds(builder, newshape, 0, 0)
    newstrides = cgutils.gep_inbounds(builder, newstrides, 0, 0)
    is_f_order = ll_intc(0)
    res = builder.call(fn, [nd, shape, strides, newnd, newshape, newstrides, ary.itemsize, is_f_order])
    return res