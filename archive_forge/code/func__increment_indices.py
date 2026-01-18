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
def _increment_indices(context, builder, ndim, shape, indices, end_flag=None, loop_continue=None, loop_break=None):
    zero = context.get_constant(types.intp, 0)
    bbend = builder.append_basic_block('end_increment')
    if end_flag is not None:
        builder.store(cgutils.false_byte, end_flag)
    for dim in reversed(range(ndim)):
        idxptr = cgutils.gep_inbounds(builder, indices, dim)
        idx = cgutils.increment_index(builder, builder.load(idxptr))
        count = shape[dim]
        in_bounds = builder.icmp_signed('<', idx, count)
        with cgutils.if_likely(builder, in_bounds):
            builder.store(idx, idxptr)
            if loop_continue is not None:
                loop_continue(dim)
            builder.branch(bbend)
        builder.store(zero, idxptr)
        if loop_break is not None:
            loop_break(dim)
    if end_flag is not None:
        builder.store(cgutils.true_byte, end_flag)
    builder.branch(bbend)
    builder.position_at_end(bbend)