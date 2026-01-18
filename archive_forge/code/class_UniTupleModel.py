from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.UniTuple)
@register_default(types.NamedUniTuple)
@register_default(types.StarArgUniTuple)
class UniTupleModel(DataModel):

    def __init__(self, dmm, fe_type):
        super(UniTupleModel, self).__init__(dmm, fe_type)
        self._elem_model = dmm.lookup(fe_type.dtype)
        self._count = len(fe_type)
        self._value_type = ir.ArrayType(self._elem_model.get_value_type(), self._count)
        self._data_type = ir.ArrayType(self._elem_model.get_data_type(), self._count)

    def get_value_type(self):
        return self._value_type

    def get_data_type(self):
        return self._data_type

    def get_return_type(self):
        return self.get_value_type()

    def get_argument_type(self):
        return (self._elem_model.get_argument_type(),) * self._count

    def as_argument(self, builder, value):
        out = []
        for i in range(self._count):
            v = builder.extract_value(value, [i])
            v = self._elem_model.as_argument(builder, v)
            out.append(v)
        return out

    def from_argument(self, builder, value):
        out = ir.Constant(self.get_value_type(), ir.Undefined)
        for i, v in enumerate(value):
            v = self._elem_model.from_argument(builder, v)
            out = builder.insert_value(out, v, [i])
        return out

    def as_data(self, builder, value):
        out = ir.Constant(self.get_data_type(), ir.Undefined)
        for i in range(self._count):
            val = builder.extract_value(value, [i])
            dval = self._elem_model.as_data(builder, val)
            out = builder.insert_value(out, dval, [i])
        return out

    def from_data(self, builder, value):
        out = ir.Constant(self.get_value_type(), ir.Undefined)
        for i in range(self._count):
            val = builder.extract_value(value, [i])
            dval = self._elem_model.from_data(builder, val)
            out = builder.insert_value(out, dval, [i])
        return out

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value

    def traverse(self, builder):

        def getter(i, value):
            return builder.extract_value(value, i)
        return [(self._fe_type.dtype, partial(getter, i)) for i in range(self._count)]

    def inner_models(self):
        return [self._elem_model]