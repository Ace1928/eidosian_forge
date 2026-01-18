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
def _equals_helper(this, other, OP):
    if not isinstance(this, types.ListType):
        return
    if not isinstance(other, types.ListType):
        return lambda this, other: False
    this_is_none = isinstance(this.dtype, types.NoneType)
    other_is_none = isinstance(other.dtype, types.NoneType)
    if this_is_none or other_is_none:

        def impl_some_none(this, other):

            def equals(this, other):
                return bool(this_is_none == other_is_none and len(this) == len(other))
            return OP(equals(this, other))
        return impl_some_none
    else:

        def impl_not_none(this, other):

            def equals(this, other):
                if len(this) != len(other):
                    return False
                for i in range(len(this)):
                    if this[i] != other[i]:
                        return False
                else:
                    return True
            return OP(equals(this, other))
        return impl_not_none