import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
class FlatSubIter(BaseSubIter):
    """
        Sub-iterator walking a contiguous array in physical order, with
        support for broadcasting (the index is reset on the outer dimension).
        """

    def init_specific(self, context, builder):
        zero = context.get_constant(types.intp, 0)
        self.set_member_ptr(cgutils.alloca_once_value(builder, zero))

    def compute_pointer(self, context, builder, indices, arrty, arr):
        index = builder.load(self.member_ptr)
        return builder.gep(arr.data, [index])

    def loop_continue(self, context, builder, logical_dim):
        if logical_dim == self.ndim - 1:
            index = builder.load(self.member_ptr)
            index = cgutils.increment_index(builder, index)
            builder.store(index, self.member_ptr)

    def loop_break(self, context, builder, logical_dim):
        if logical_dim == 0:
            zero = context.get_constant(types.intp, 0)
            builder.store(zero, self.member_ptr)
        elif logical_dim == self.ndim - 1:
            index = builder.load(self.member_ptr)
            index = cgutils.increment_index(builder, index)
            builder.store(index, self.member_ptr)