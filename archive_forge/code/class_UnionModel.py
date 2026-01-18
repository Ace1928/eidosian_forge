from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.UnionType)
class UnionModel(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('tag', types.uintp), ('payload', types.Tuple.from_types(fe_type.types))]
        super(UnionModel, self).__init__(dmm, fe_type, members)