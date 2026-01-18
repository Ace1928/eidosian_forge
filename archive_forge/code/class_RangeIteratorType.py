from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.RangeIteratorType)
class RangeIteratorType(StructModel):

    def __init__(self, dmm, fe_type):
        int_type = fe_type.yield_type
        members = [('iter', types.EphemeralPointer(int_type)), ('stop', int_type), ('step', int_type), ('count', types.EphemeralPointer(int_type))]
        super(RangeIteratorType, self).__init__(dmm, fe_type, members)