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
def _change_dtype(context, builder, oldty, newty, ary):
    """
    Attempt to fix up *ary* for switching from *oldty* to *newty*.

    See Numpy's array_descr_set()
    (np/core/src/multiarray/getset.c).
    Attempt to fix the array's shape and strides for a new dtype.
    False is returned on failure, True on success.
    """
    assert oldty.ndim == newty.ndim
    assert oldty.layout == newty.layout
    new_layout = ord(newty.layout)
    any_layout = ord('A')
    c_layout = ord('C')
    f_layout = ord('F')
    int8 = types.int8

    def imp(nd, dims, strides, old_itemsize, new_itemsize, layout):
        if layout == any_layout:
            if strides[-1] == old_itemsize:
                layout = int8(c_layout)
            elif strides[0] == old_itemsize:
                layout = int8(f_layout)
        if old_itemsize != new_itemsize and (layout == any_layout or nd == 0):
            return False
        if layout == c_layout:
            i = nd - 1
        else:
            i = 0
        if new_itemsize < old_itemsize:
            if old_itemsize % new_itemsize != 0:
                return False
            newdim = old_itemsize // new_itemsize
            dims[i] *= newdim
            strides[i] = new_itemsize
        elif new_itemsize > old_itemsize:
            bytelength = dims[i] * old_itemsize
            if bytelength % new_itemsize != 0:
                return False
            dims[i] = bytelength // new_itemsize
            strides[i] = new_itemsize
        else:
            pass
        return True
    old_itemsize = context.get_constant(types.intp, get_itemsize(context, oldty))
    new_itemsize = context.get_constant(types.intp, get_itemsize(context, newty))
    nd = context.get_constant(types.intp, newty.ndim)
    shape_data = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('shape'), 0, 0)
    strides_data = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('strides'), 0, 0)
    shape_strides_array_type = types.Array(dtype=types.intp, ndim=1, layout='C')
    arycls = context.make_array(shape_strides_array_type)
    shape_constant = cgutils.pack_array(builder, [context.get_constant(types.intp, newty.ndim)])
    sizeof_intp = context.get_abi_sizeof(context.get_data_type(types.intp))
    sizeof_intp = context.get_constant(types.intp, sizeof_intp)
    strides_constant = cgutils.pack_array(builder, [sizeof_intp])
    shape_ary = arycls(context, builder)
    populate_array(shape_ary, data=shape_data, shape=shape_constant, strides=strides_constant, itemsize=sizeof_intp, meminfo=None)
    strides_ary = arycls(context, builder)
    populate_array(strides_ary, data=strides_data, shape=shape_constant, strides=strides_constant, itemsize=sizeof_intp, meminfo=None)
    shape = shape_ary._getvalue()
    strides = strides_ary._getvalue()
    args = [nd, shape, strides, old_itemsize, new_itemsize, context.get_constant(types.int8, new_layout)]
    sig = signature(types.boolean, types.intp, shape_strides_array_type, shape_strides_array_type, types.intp, types.intp, types.int8)
    res = context.compile_internal(builder, imp, sig, args)
    update_array_info(newty, ary)
    res = impl_ret_borrowed(context, builder, sig.return_type, res)
    return res