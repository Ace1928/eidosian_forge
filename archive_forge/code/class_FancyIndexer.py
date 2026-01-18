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
class FancyIndexer(object):
    """
    Perform fancy indexing on the given array.
    """

    def __init__(self, context, builder, aryty, ary, index_types, indices):
        self.context = context
        self.builder = builder
        self.aryty = aryty
        self.shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)
        self.strides = cgutils.unpack_tuple(builder, ary.strides, aryty.ndim)
        self.ll_intp = self.context.get_value_type(types.intp)
        self.newaxes = []
        indexers = []
        num_newaxes = len([idx for idx in index_types if is_nonelike(idx)])
        ax = 0
        new_ax = 0
        for indexval, idxty in zip(indices, index_types):
            if idxty is types.ellipsis:
                n_missing = aryty.ndim - len(indices) + 1 + num_newaxes
                for i in range(n_missing):
                    indexer = EntireIndexer(context, builder, aryty, ary, ax)
                    indexers.append(indexer)
                    ax += 1
                    new_ax += 1
                continue
            if isinstance(idxty, types.SliceType):
                slice = context.make_helper(builder, idxty, indexval)
                indexer = SliceIndexer(context, builder, aryty, ary, ax, idxty, slice)
                indexers.append(indexer)
            elif isinstance(idxty, types.Integer):
                ind = fix_integer_index(context, builder, idxty, indexval, self.shapes[ax])
                indexer = IntegerIndexer(context, builder, ind)
                indexers.append(indexer)
            elif isinstance(idxty, types.Array):
                idxary = make_array(idxty)(context, builder, indexval)
                if isinstance(idxty.dtype, types.Integer):
                    indexer = IntegerArrayIndexer(context, builder, idxty, idxary, self.shapes[ax])
                elif isinstance(idxty.dtype, types.Boolean):
                    indexer = BooleanArrayIndexer(context, builder, idxty, idxary)
                else:
                    assert 0
                indexers.append(indexer)
            elif is_nonelike(idxty):
                self.newaxes.append(new_ax)
                ax -= 1
            else:
                raise AssertionError('unexpected index type: %s' % (idxty,))
            ax += 1
            new_ax += 1
        assert ax <= aryty.ndim, (ax, aryty.ndim)
        while ax < aryty.ndim:
            indexer = EntireIndexer(context, builder, aryty, ary, ax)
            indexers.append(indexer)
            ax += 1
        assert len(indexers) == aryty.ndim, (len(indexers), aryty.ndim)
        self.indexers = indexers

    def prepare(self):
        for i in self.indexers:
            i.prepare()
        one = self.context.get_constant(types.intp, 1)
        res_shape = [i.get_shape() for i in self.indexers]
        for i in self.newaxes:
            res_shape.insert(i, (one,))
        self.indexers_shape = sum(res_shape, ())

    def get_shape(self):
        """
        Get the resulting data shape as Python tuple.
        """
        return self.indexers_shape

    def get_offset_bounds(self, strides, itemsize):
        """
        Get a half-open [lower, upper) range of byte offsets spanned by
        the indexer with the given strides and itemsize.  The indexer is
        guaranteed to not go past those bounds.
        """
        assert len(strides) == self.aryty.ndim
        builder = self.builder
        is_empty = cgutils.false_bit
        zero = self.ll_intp(0)
        one = self.ll_intp(1)
        lower = zero
        upper = zero
        for indexer, shape, stride in zip(self.indexers, self.indexers_shape, strides):
            is_empty = builder.or_(is_empty, builder.icmp_unsigned('==', shape, zero))
            lower_index, upper_index = indexer.get_index_bounds()
            lower_offset = builder.mul(stride, lower_index)
            upper_offset = builder.mul(stride, builder.sub(upper_index, one))
            is_downwards = builder.icmp_signed('<', stride, zero)
            lower = builder.add(lower, builder.select(is_downwards, upper_offset, lower_offset))
            upper = builder.add(upper, builder.select(is_downwards, lower_offset, upper_offset))
        upper = builder.add(upper, itemsize)
        lower = builder.select(is_empty, zero, lower)
        upper = builder.select(is_empty, zero, upper)
        return (lower, upper)

    def begin_loops(self):
        indices, counts = zip(*(i.loop_head() for i in self.indexers))
        return (indices, counts)

    def end_loops(self):
        for i in reversed(self.indexers):
            i.loop_tail()