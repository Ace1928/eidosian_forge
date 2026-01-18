import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def _dot2_codegen(context, builder, sig, args):
    ensure_blas()
    with make_contiguous(context, builder, sig, args) as (sig, args):
        if ndims == (2, 2):
            return dot_2_mm(context, builder, sig, args)
        elif ndims == (2, 1):
            return dot_2_mv(context, builder, sig, args)
        elif ndims == (1, 2):
            return dot_2_vm(context, builder, sig, args)
        elif ndims == (1, 1):
            return dot_2_vv(context, builder, sig, args)
        else:
            raise AssertionError('unreachable')