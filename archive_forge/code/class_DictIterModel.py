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
@register_model(DictItemsIterableType)
@register_model(DictKeysIterableType)
@register_model(DictValuesIterableType)
@register_model(DictIteratorType)
class DictIterModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('parent', fe_type.parent), ('state', types.voidptr)]
        super(DictIterModel, self).__init__(dmm, fe_type, members)