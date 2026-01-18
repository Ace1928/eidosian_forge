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
def _dict_dump(typingctx, d):
    """Dump the dictionary keys and values.
    Wraps numba_dict_dump for debugging.
    """
    resty = types.void
    sig = resty(d)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ir.VoidType(), [ll_dict_type])
        [td] = sig.args
        [d] = args
        dp = _container_get_data(context, builder, td, d)
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_dump')
        builder.call(fn, [dp])
    return (sig, codegen)