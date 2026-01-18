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
def _dict_insert(typingctx, d, key, hashval, val):
    """Wrap numba_dict_insert
    """
    resty = types.int32
    sig = resty(d, d.key_type, types.intp, d.value_type)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_status, [ll_dict_type, ll_bytes, ll_hash, ll_bytes, ll_bytes])
        [d, key, hashval, val] = args
        [td, tkey, thashval, tval] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_insert')
        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[tval]
        data_key = dm_key.as_data(builder, key)
        data_val = dm_val.as_data(builder, val)
        ptr_key = cgutils.alloca_once_value(builder, data_key)
        cgutils.memset_padding(builder, ptr_key)
        ptr_val = cgutils.alloca_once_value(builder, data_val)
        ptr_oldval = cgutils.alloca_once(builder, data_val.type)
        dp = _container_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, _as_bytes(builder, ptr_key), hashval, _as_bytes(builder, ptr_val), _as_bytes(builder, ptr_oldval)])
        return status
    return (sig, codegen)