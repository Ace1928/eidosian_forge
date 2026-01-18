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
def _dict_popitem(typingctx, d):
    """Wrap numba_dict_popitem
    """
    keyvalty = types.Tuple([d.key_type, d.value_type])
    resty = types.Tuple([types.int32, types.Optional(keyvalty)])
    sig = resty(d)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_status, [ll_dict_type, ll_bytes, ll_bytes])
        [d] = args
        [td] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_popitem')
        dm_key = context.data_model_manager[td.key_type]
        dm_val = context.data_model_manager[td.value_type]
        ptr_key = cgutils.alloca_once(builder, dm_key.get_data_type())
        ptr_val = cgutils.alloca_once(builder, dm_val.get_data_type())
        dp = _container_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, _as_bytes(builder, ptr_key), _as_bytes(builder, ptr_val)])
        out = context.make_optional_none(builder, keyvalty)
        pout = cgutils.alloca_once_value(builder, out)
        cond = builder.icmp_signed('==', status, status.type(int(Status.OK)))
        with builder.if_then(cond):
            key = dm_key.load_from_data_pointer(builder, ptr_key)
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            keyval = context.make_tuple(builder, keyvalty, [key, val])
            optkeyval = context.make_optional_value(builder, keyvalty, keyval)
            builder.store(optkeyval, pout)
        out = builder.load(pout)
        return cgutils.pack_struct(builder, [status, out])
    return (sig, codegen)