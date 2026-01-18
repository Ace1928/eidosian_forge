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
def _as_layout_array(context, builder, sig, args, output_layout):
    """
    Common logic for layout conversion function;
    e.g. ascontiguousarray and asfortranarray
    """
    retty = sig.return_type
    aryty = sig.args[0]
    assert retty.layout == output_layout, 'return-type has incorrect layout'
    if aryty.ndim == 0:
        assert retty.ndim == 1
        ary = make_array(aryty)(context, builder, value=args[0])
        ret = make_array(retty)(context, builder)
        shape = context.get_constant_generic(builder, types.UniTuple(types.intp, 1), (1,))
        strides = context.make_tuple(builder, types.UniTuple(types.intp, 1), (ary.itemsize,))
        populate_array(ret, ary.data, shape, strides, ary.itemsize, ary.meminfo, ary.parent)
        return impl_ret_borrowed(context, builder, retty, ret._getvalue())
    elif retty.layout == aryty.layout or (aryty.ndim == 1 and aryty.layout in 'CF'):
        return impl_ret_borrowed(context, builder, retty, args[0])
    elif aryty.layout == 'A':
        assert output_layout in 'CF'
        check_func = is_contiguous if output_layout == 'C' else is_fortran
        is_contig = _call_contiguous_check(check_func, context, builder, aryty, args[0])
        with builder.if_else(is_contig) as (then, orelse):
            with then:
                out_then = impl_ret_borrowed(context, builder, retty, args[0])
                then_blk = builder.block
            with orelse:
                out_orelse = _array_copy(context, builder, sig, args)
                orelse_blk = builder.block
        ret_phi = builder.phi(out_then.type)
        ret_phi.add_incoming(out_then, then_blk)
        ret_phi.add_incoming(out_orelse, orelse_blk)
        return ret_phi
    else:
        return _array_copy(context, builder, sig, args)