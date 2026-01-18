import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
class ListIterInstance(_ListPayloadMixin):

    def __init__(self, context, builder, iter_type, iter_val):
        self._context = context
        self._builder = builder
        self._ty = iter_type
        self._iter = context.make_helper(builder, iter_type, iter_val)
        self._datamodel = context.data_model_manager[iter_type.yield_type]

    @classmethod
    def from_list(cls, context, builder, iter_type, list_val):
        list_inst = ListInstance(context, builder, iter_type.container, list_val)
        self = cls(context, builder, iter_type, None)
        index = context.get_constant(types.intp, 0)
        self._iter.index = cgutils.alloca_once_value(builder, index)
        self._iter.meminfo = list_inst.meminfo
        return self

    @property
    def _payload(self):
        return get_list_payload(self._context, self._builder, self._ty.container, self._iter)

    @property
    def value(self):
        return self._iter._getvalue()

    @property
    def index(self):
        return self._builder.load(self._iter.index)

    @index.setter
    def index(self, value):
        self._builder.store(value, self._iter.index)