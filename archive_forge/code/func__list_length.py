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
def _list_length(typingctx, l):
    """Wrap numba_list_length

    Returns the length of the list.
    """
    sig = types.intp(l)

    def codegen(context, builder, sig, args):
        [tl] = sig.args
        [l] = args
        fnty = ir.FunctionType(ll_ssize_t, [ll_list_type])
        fname = 'numba_list_size_address'
        fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
        fn.attributes.add('alwaysinline')
        fn.attributes.add('readonly')
        fn.attributes.add('nounwind')
        lp = _container_get_data(context, builder, tl, l)
        len_addr = builder.call(fn, [lp])
        ptr = builder.inttoptr(len_addr, cgutils.intp_t.as_pointer())
        return builder.load(ptr)
    return (sig, codegen)