from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class FlatIter(StructModel):

    def __init__(self, dmm, fe_type):
        array_type = fe_type.array_type
        dtype = array_type.dtype
        ndim = array_type.ndim
        members = [('array', array_type), ('pointers', types.EphemeralArray(types.CPointer(dtype), ndim)), ('indices', types.EphemeralArray(types.intp, ndim)), ('exhausted', types.EphemeralPointer(types.boolean))]
        super(FlatIter, self).__init__(dmm, fe_type, members)