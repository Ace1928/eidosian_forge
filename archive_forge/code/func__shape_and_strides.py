from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _shape_and_strides(self, context, builder):
    one = context.get_constant(types.intp, 1)
    zero = context.get_constant(types.intp, 0)
    shape = cgutils.pack_array(builder, [one])
    strides = cgutils.pack_array(builder, [zero])
    return (shape, strides)