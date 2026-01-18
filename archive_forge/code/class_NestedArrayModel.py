from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.NestedArray)
class NestedArrayModel(ArrayModel):

    def __init__(self, dmm, fe_type):
        self._be_type = dmm.lookup(fe_type.dtype).get_data_type()
        super(NestedArrayModel, self).__init__(dmm, fe_type)

    def as_storage_type(self):
        """Return the LLVM type representation for the storage of
        the nestedarray.
        """
        ret = ir.ArrayType(self._be_type, self._fe_type.nitems)
        return ret