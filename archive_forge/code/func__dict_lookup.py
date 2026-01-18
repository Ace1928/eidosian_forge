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
def _dict_lookup(typingctx, d, key, hashval):
    """Wrap numba_dict_lookup

    Returns 2-tuple of (intp, ?value_type)
    """
    resty = types.Tuple([types.intp, types.Optional(d.value_type)])
    sig = resty(d, key, hashval)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_ssize_t, [ll_dict_type, ll_bytes, ll_hash, ll_bytes])
        [td, tkey, thashval] = sig.args
        [d, key, hashval] = args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_lookup')
        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[td.value_type]
        data_key = dm_key.as_data(builder, key)
        ptr_key = cgutils.alloca_once_value(builder, data_key)
        cgutils.memset_padding(builder, ptr_key)
        ll_val = context.get_data_type(td.value_type)
        ptr_val = cgutils.alloca_once(builder, ll_val)
        dp = _container_get_data(context, builder, td, d)
        ix = builder.call(fn, [dp, _as_bytes(builder, ptr_key), hashval, _as_bytes(builder, ptr_val)])
        found = builder.icmp_signed('>', ix, ix.type(int(DKIX.EMPTY)))
        out = context.make_optional_none(builder, td.value_type)
        pout = cgutils.alloca_once_value(builder, out)
        with builder.if_then(found):
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            context.nrt.incref(builder, td.value_type, val)
            loaded = context.make_optional_value(builder, td.value_type, val)
            builder.store(loaded, pout)
        out = builder.load(pout)
        return context.make_tuple(builder, resty, [ix, out])
    return (sig, codegen)