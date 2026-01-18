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
def _broadcast_onto(src_ndim, src_shape, dest_ndim, dest_shape):
    """Low-level utility function used in calculating a shape for
    an implicit output array.  This function assumes that the
    destination shape is an LLVM pointer to a C-style array that was
    already initialized to a size of one along all axes.

    Returns an integer value:
    >= 1  :  Succeeded.  Return value should equal the number of dimensions in
             the destination shape.
    0     :  Failed to broadcast because source shape is larger than the
             destination shape (this case should be weeded out at type
             checking).
    < 0   :  Failed to broadcast onto destination axis, at axis number ==
             -(return_value + 1).
    """
    if src_ndim > dest_ndim:
        return 0
    else:
        src_index = 0
        dest_index = dest_ndim - src_ndim
        while src_index < src_ndim:
            src_dim_size = src_shape[src_index]
            dest_dim_size = dest_shape[dest_index]
            if dest_dim_size != 1:
                if src_dim_size != dest_dim_size and src_dim_size != 1:
                    return -(dest_index + 1)
            elif src_dim_size != 1:
                dest_shape[dest_index] = src_dim_size
            src_index += 1
            dest_index += 1
    return dest_index