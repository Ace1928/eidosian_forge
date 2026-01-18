from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
def _guess_dtype(iterable):
    """Guess the correct dtype of the iterable type. """
    if not isinstance(iterable, types.IterableType):
        raise TypingError('List() argument must be iterable')
    elif isinstance(iterable, types.Array) and iterable.ndim > 1:
        return iterable.copy(ndim=iterable.ndim - 1, layout='A')
    elif hasattr(iterable, 'dtype'):
        return iterable.dtype
    elif hasattr(iterable, 'yield_type'):
        return iterable.yield_type
    elif isinstance(iterable, types.UnicodeType):
        return iterable
    elif isinstance(iterable, types.DictType):
        return iterable.key_type
    else:
        raise TypingError('List() argument does not have a suitable dtype')