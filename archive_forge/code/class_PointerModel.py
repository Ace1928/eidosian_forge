from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.CPointer)
class PointerModel(PrimitiveModel):

    def __init__(self, dmm, fe_type):
        self._pointee_model = dmm.lookup(fe_type.dtype)
        self._pointee_be_type = self._pointee_model.get_data_type()
        be_type = self._pointee_be_type.as_pointer()
        super(PointerModel, self).__init__(dmm, fe_type, be_type)