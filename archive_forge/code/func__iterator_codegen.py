import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
def _iterator_codegen(resty):
    """The common codegen for iterator intrinsics.

    Populates the iterator struct and increfs.
    """

    def codegen(context, builder, sig, args):
        [d] = args
        [td] = sig.args
        iterhelper = context.make_helper(builder, resty)
        iterhelper.parent = d
        iterhelper.state = iterhelper.state.type(None)
        return impl_ret_borrowed(context, builder, resty, iterhelper._getvalue())
    return codegen