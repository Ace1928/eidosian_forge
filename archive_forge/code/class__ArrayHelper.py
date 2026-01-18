import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import (
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
class _ArrayHelper(namedtuple('_ArrayHelper', ('context', 'builder', 'shape', 'strides', 'data', 'layout', 'base_type', 'ndim', 'return_val'))):
    """Helper class to handle array arguments/result.
    It provides methods to generate code loading/storing specific
    items as well as support code for handling indices.
    """

    def create_iter_indices(self):
        intpty = self.context.get_value_type(types.intp)
        ZERO = ir.Constant(ir.IntType(intpty.width), 0)
        indices = []
        for i in range(self.ndim):
            x = cgutils.alloca_once(self.builder, ir.IntType(intpty.width))
            self.builder.store(ZERO, x)
            indices.append(x)
        return _ArrayIndexingHelper(self, indices)

    def _load_effective_address(self, indices):
        return cgutils.get_item_pointer2(self.context, self.builder, data=self.data, shape=self.shape, strides=self.strides, layout=self.layout, inds=indices)

    def load_data(self, indices):
        model = self.context.data_model_manager[self.base_type]
        ptr = self._load_effective_address(indices)
        return model.load_from_data_pointer(self.builder, ptr)

    def store_data(self, indices, value):
        ctx = self.context
        bld = self.builder
        store_value = ctx.get_value_as_data(bld, self.base_type, value)
        assert ctx.get_data_type(self.base_type) == store_value.type
        bld.store(store_value, self._load_effective_address(indices))