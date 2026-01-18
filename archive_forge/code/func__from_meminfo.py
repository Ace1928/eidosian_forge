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
def _from_meminfo(typingctx, mi, dicttyperef):
    """Recreate a dictionary from a MemInfoPointer
    """
    if mi != _meminfo_dictptr:
        raise TypingError('expected a MemInfoPointer for dict.')
    dicttype = dicttyperef.instance_type
    if not isinstance(dicttype, DictType):
        raise TypingError('expected a {}'.format(DictType))

    def codegen(context, builder, sig, args):
        [tmi, tdref] = sig.args
        td = tdref.instance_type
        [mi, _] = args
        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder)
        data_pointer = context.nrt.meminfo_data(builder, mi)
        data_pointer = builder.bitcast(data_pointer, ll_dict_type.as_pointer())
        dstruct.data = builder.load(data_pointer)
        dstruct.meminfo = mi
        return impl_ret_borrowed(context, builder, dicttype, dstruct._getvalue())
    sig = dicttype(mi, dicttyperef)
    return (sig, codegen)