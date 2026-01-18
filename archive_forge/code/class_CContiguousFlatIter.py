from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class CContiguousFlatIter(StructModel):

    def __init__(self, dmm, fe_type, need_indices):
        assert fe_type.array_type.layout == 'C'
        array_type = fe_type.array_type
        dtype = array_type.dtype
        ndim = array_type.ndim
        members = [('array', array_type), ('stride', types.intp), ('index', types.EphemeralPointer(types.intp))]
        if need_indices:
            members.append(('indices', types.EphemeralArray(types.intp, ndim)))
        super(CContiguousFlatIter, self).__init__(dmm, fe_type, members)