from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.UnicodeCharSeq)
class UnicodeCharSeq(DataModel):

    def __init__(self, dmm, fe_type):
        super(UnicodeCharSeq, self).__init__(dmm, fe_type)
        charty = ir.IntType(numpy_support.sizeof_unicode_char * 8)
        self._be_type = ir.ArrayType(charty, fe_type.count)

    def get_value_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type

    def as_data(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value