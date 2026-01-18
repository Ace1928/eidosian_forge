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
@intrinsic
def _dict_length(typingctx, d):
    """Wrap numba_dict_length

    Returns the length of the dictionary.
    """
    resty = types.intp
    sig = resty(d)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_ssize_t, [ll_dict_type])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_length')
        [d] = args
        [td] = sig.args
        dp = _container_get_data(context, builder, td, d)
        n = builder.call(fn, [dp])
        return n
    return (sig, codegen)