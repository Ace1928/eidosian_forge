from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.StructRefPayload)
class StructPayloadModel(StructModel):
    """Model for the payload of a mutable struct
    """

    def __init__(self, dmm, fe_typ):
        members = tuple(fe_typ.field_dict.items())
        super().__init__(dmm, fe_typ, members)