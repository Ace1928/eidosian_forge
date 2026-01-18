from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class _ArrayAsScalarArgLoader(_ArrayArgLoader):
    """
    Handle GUFunc argument loading where the shape signature specifies
    a scalar "()" but a 1D array is used for the type of the core function.
    """

    def _shape_and_strides(self, context, builder):
        one = context.get_constant(types.intp, 1)
        zero = context.get_constant(types.intp, 0)
        shape = cgutils.pack_array(builder, [one])
        strides = cgutils.pack_array(builder, [zero])
        return (shape, strides)