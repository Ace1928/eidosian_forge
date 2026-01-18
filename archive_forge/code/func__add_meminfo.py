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
def _add_meminfo(context, builder, lstruct):
    alloc_size = context.get_abi_sizeof(context.get_value_type(types.voidptr))
    dtor = _imp_dtor(context, builder.module)
    meminfo = context.nrt.meminfo_alloc_dtor(builder, context.get_constant(types.uintp, alloc_size), dtor)
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer, ll_list_type.as_pointer())
    builder.store(lstruct.data, data_pointer)
    lstruct.meminfo = meminfo