from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class _ArrayArgLoader(object):
    """
    Handle GUFunc argument loading where an array is expected.
    """

    def __init__(self, dtype, ndim, core_step, as_scalar, shape, strides):
        self.dtype = dtype
        self.ndim = ndim
        self.core_step = core_step
        self.as_scalar = as_scalar
        self.shape = shape
        self.strides = strides

    def load(self, context, builder, data, ind):
        arytyp = types.Array(dtype=self.dtype, ndim=self.ndim, layout='A')
        arycls = context.make_array(arytyp)
        array = arycls(context, builder)
        offseted_data = cgutils.pointer_add(builder, data, builder.mul(self.core_step, ind))
        shape, strides = self._shape_and_strides(context, builder)
        itemsize = context.get_abi_sizeof(context.get_data_type(self.dtype))
        context.populate_array(array, data=builder.bitcast(offseted_data, array.data.type), shape=shape, strides=strides, itemsize=context.get_constant(types.intp, itemsize), meminfo=None)
        return array._getvalue()

    def _shape_and_strides(self, context, builder):
        shape = cgutils.pack_array(builder, self.shape)
        strides = cgutils.pack_array(builder, self.strides)
        return (shape, strides)