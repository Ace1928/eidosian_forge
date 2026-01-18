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
class ListStatus(IntEnum):
    """Status code for other list operations.
    """
    LIST_OK = (0,)
    LIST_ERR_INDEX = -1
    LIST_ERR_NO_MEMORY = -2
    LIST_ERR_MUTATED = -3
    LIST_ERR_ITER_EXHAUSTED = -4
    LIST_ERR_IMMUTABLE = -5