from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.SetEntry)
class SetEntryModel(StructModel):

    def __init__(self, dmm, fe_type):
        dtype = fe_type.set_type.dtype
        members = [('hash', types.intp), ('key', dtype)]
        super(SetEntryModel, self).__init__(dmm, fe_type, members)