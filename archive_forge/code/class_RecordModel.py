from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Record)
class RecordModel(CompositeModel):

    def __init__(self, dmm, fe_type):
        super(RecordModel, self).__init__(dmm, fe_type)
        self._models = [self._dmm.lookup(t) for _, t in fe_type.members]
        self._be_type = ir.ArrayType(ir.IntType(8), fe_type.size)
        self._be_ptr_type = self._be_type.as_pointer()

    def get_value_type(self):
        """Passed around as reference to underlying data
        """
        return self._be_ptr_type

    def get_argument_type(self):
        return self._be_ptr_type

    def get_return_type(self):
        return self._be_ptr_type

    def get_data_type(self):
        return self._be_type

    def as_data(self, builder, value):
        return builder.load(value)

    def from_data(self, builder, value):
        raise NotImplementedError('use load_from_data_pointer() instead')

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value

    def load_from_data_pointer(self, builder, ptr, align=None):
        return builder.bitcast(ptr, self.get_value_type())