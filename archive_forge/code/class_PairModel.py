from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Pair)
class PairModel(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('first', fe_type.first_type), ('second', fe_type.second_type)]
        super(PairModel, self).__init__(dmm, fe_type, members)