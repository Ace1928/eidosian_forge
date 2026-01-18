from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.EphemeralPointer)
class EphemeralPointerModel(PointerModel):

    def get_data_type(self):
        return self._pointee_be_type

    def as_data(self, builder, value):
        value = builder.load(value)
        return self._pointee_model.as_data(builder, value)

    def from_data(self, builder, value):
        raise NotImplementedError('use load_from_data_pointer() instead')

    def load_from_data_pointer(self, builder, ptr, align=None):
        return builder.bitcast(ptr, self.get_value_type())