from __future__ import annotations
import operator
import threading
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import ValuesView
import weakref
from ._has_cy import HAS_CYEXTENSION
from .typing import is_non_string_iterable
from .typing import Literal
from .typing import Protocol
class ThreadLocalRegistry(ScopedRegistry[_T]):
    """A :class:`.ScopedRegistry` that uses a ``threading.local()``
    variable for storage.

    """

    def __init__(self, createfunc: Callable[[], _T]):
        self.createfunc = createfunc
        self.registry = threading.local()

    def __call__(self) -> _T:
        try:
            return self.registry.value
        except AttributeError:
            val = self.registry.value = self.createfunc()
            return val

    def has(self) -> bool:
        return hasattr(self.registry, 'value')

    def set(self, obj: _T) -> None:
        self.registry.value = obj

    def clear(self) -> None:
        try:
            del self.registry.value
        except AttributeError:
            pass