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
@register_jitable
def _can_broadcast(array, dest_shape):
    src_shape = array.shape
    src_ndim = len(src_shape)
    dest_ndim = len(dest_shape)
    if src_ndim > dest_ndim:
        raise ValueError('input operand has more dimensions than allowed by the axis remapping')
    for size in dest_shape:
        if size < 0:
            raise ValueError('all elements of broadcast shape must be non-negative')
    src_index = 0
    dest_index = dest_ndim - src_ndim
    while src_index < src_ndim:
        src_dim = src_shape[src_index]
        dest_dim = dest_shape[dest_index]
        if src_dim == dest_dim or src_dim == 1:
            src_index += 1
            dest_index += 1
        else:
            raise ValueError('operands could not be broadcast together with remapped shapes')