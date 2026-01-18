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
class IntegerIndexer(Indexer):
    """
    Compute indices from a single integer.
    """

    def __init__(self, context, builder, idx):
        self.context = context
        self.builder = builder
        self.idx = idx
        self.ll_intp = self.context.get_value_type(types.intp)

    def prepare(self):
        pass

    def get_size(self):
        return Constant(self.ll_intp, 1)

    def get_shape(self):
        return ()

    def get_index_bounds(self):
        return (self.idx, self.builder.add(self.idx, self.get_size()))

    def loop_head(self):
        return (self.idx, None)

    def loop_tail(self):
        pass