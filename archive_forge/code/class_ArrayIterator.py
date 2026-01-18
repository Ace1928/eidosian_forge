from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.ArrayIterator)
class ArrayIterator(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('index', types.EphemeralPointer(types.uintp)), ('array', fe_type.array_type)]
        super(ArrayIterator, self).__init__(dmm, fe_type, members)