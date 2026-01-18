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
class IntegerArrayIndexer(Indexer):
    """
    Compute indices from an array of integer indices.
    """

    def __init__(self, context, builder, idxty, idxary, size):
        self.context = context
        self.builder = builder
        self.idxty = idxty
        self.idxary = idxary
        self.size = size
        assert idxty.ndim == 1
        self.ll_intp = self.context.get_value_type(types.intp)

    def prepare(self):
        builder = self.builder
        self.idx_size = cgutils.unpack_tuple(builder, self.idxary.shape)[0]
        self.idx_index = cgutils.alloca_once(builder, self.ll_intp)
        self.bb_start = builder.append_basic_block()
        self.bb_end = builder.append_basic_block()

    def get_size(self):
        return self.idx_size

    def get_shape(self):
        return (self.idx_size,)

    def get_index_bounds(self):
        return (self.ll_intp(0), self.size)

    def loop_head(self):
        builder = self.builder
        self.builder.store(Constant(self.ll_intp, 0), self.idx_index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_start)
        cur_index = builder.load(self.idx_index)
        with builder.if_then(builder.icmp_signed('>=', cur_index, self.idx_size), likely=False):
            builder.branch(self.bb_end)
        index = _getitem_array_single_int(self.context, builder, self.idxty.dtype, self.idxty, self.idxary, cur_index)
        index = fix_integer_index(self.context, builder, self.idxty.dtype, index, self.size)
        return (index, cur_index)

    def loop_tail(self):
        builder = self.builder
        next_index = cgutils.increment_index(builder, builder.load(self.idx_index))
        builder.store(next_index, self.idx_index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_end)