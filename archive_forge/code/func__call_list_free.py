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
def _call_list_free(context, builder, ptr):
    """Call numba_list_free(ptr)
    """
    fnty = ir.FunctionType(ir.VoidType(), [ll_list_type])
    free = cgutils.get_or_insert_function(builder.module, fnty, 'numba_list_free')
    builder.call(free, [ptr])