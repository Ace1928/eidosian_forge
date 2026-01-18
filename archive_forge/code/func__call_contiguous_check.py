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
def _call_contiguous_check(checker, context, builder, aryty, ary):
    """Helper to invoke the contiguous checker function on an array

    Args
    ----
    checker :
        ``numba.numpy_supports.is_contiguous``, or
        ``numba.numpy_supports.is_fortran``.
    context : target context
    builder : llvm ir builder
    aryty : numba type
    ary : llvm value
    """
    ary = make_array(aryty)(context, builder, value=ary)
    tup_intp = types.UniTuple(types.intp, aryty.ndim)
    itemsize = context.get_abi_sizeof(context.get_value_type(aryty.dtype))
    check_sig = signature(types.bool_, tup_intp, tup_intp, types.intp)
    check_args = [ary.shape, ary.strides, context.get_constant(types.intp, itemsize)]
    is_contig = context.compile_internal(builder, checker, check_sig, check_args)
    return is_contig