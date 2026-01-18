import operator
from enum import IntEnum
from llvmlite import ir
from numba.core.extending import (
from numba.core.imputils import iternext_impl
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
from numba.cpython import listobj
@intrinsic
def _list_setitem(typingctx, l, index, item):
    """Wrap numba_list_setitem
    """
    resty = types.int32
    sig = resty(l, index, item)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_status, [ll_list_type, ll_ssize_t, ll_bytes])
        [l, index, item] = args
        [tl, tindex, titem] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_list_setitem')
        dm_item = context.data_model_manager[titem]
        data_item = dm_item.as_data(builder, item)
        ptr_item = cgutils.alloca_once_value(builder, data_item)
        lp = _container_get_data(context, builder, tl, l)
        status = builder.call(fn, [lp, index, _as_bytes(builder, ptr_item)])
        return status
    return (sig, codegen)