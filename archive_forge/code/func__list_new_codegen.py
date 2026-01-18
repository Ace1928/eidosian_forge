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
def _list_new_codegen(context, builder, itemty, new_size, error_handler):
    fnty = ir.FunctionType(ll_status, [ll_list_type.as_pointer(), ll_ssize_t, ll_ssize_t])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_list_new')
    ll_item = context.get_data_type(itemty)
    sz_item = context.get_abi_sizeof(ll_item)
    reflp = cgutils.alloca_once(builder, ll_list_type, zfill=True)
    status = builder.call(fn, [reflp, ll_ssize_t(sz_item), new_size])
    msg = 'Failed to allocate list'
    error_handler(builder, status, msg)
    lp = builder.load(reflp)
    return lp