from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.SetPayload)
class SetPayloadModel(StructModel):

    def __init__(self, dmm, fe_type):
        entry_type = types.SetEntry(fe_type.container)
        members = [('fill', types.intp), ('used', types.intp), ('mask', types.intp), ('finger', types.intp), ('dirty', types.boolean), ('entries', entry_type)]
        super(SetPayloadModel, self).__init__(dmm, fe_type, members)