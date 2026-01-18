from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.misc.SliceLiteral)
@register_default(types.SliceType)
class SliceModel(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('start', types.intp), ('stop', types.intp), ('step', types.intp)]
        super(SliceModel, self).__init__(dmm, fe_type, members)