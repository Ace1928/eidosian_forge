from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
class HasMemoized:
    """A mixin class that maintains the names of memoized elements in a
    collection for easy cache clearing, generative, etc.

    """
    if not TYPE_CHECKING:
        __slots__ = ()
    _memoized_keys: FrozenSet[str] = frozenset()

    def _reset_memoizations(self) -> None:
        for elem in self._memoized_keys:
            self.__dict__.pop(elem, None)

    def _assert_no_memoizations(self) -> None:
        for elem in self._memoized_keys:
            assert elem not in self.__dict__

    def _set_memoized_attribute(self, key: str, value: Any) -> None:
        self.__dict__[key] = value
        self._memoized_keys |= {key}

    class memoized_attribute(memoized_property[_T]):
        """A read-only @property that is only evaluated once.

        :meta private:

        """
        fget: Callable[..., _T]
        __doc__: Optional[str]
        __name__: str

        def __init__(self, fget: Callable[..., _T], doc: Optional[str]=None):
            self.fget = fget
            self.__doc__ = doc or fget.__doc__
            self.__name__ = fget.__name__

        @overload
        def __get__(self: _MA, obj: None, cls: Any) -> _MA:
            ...

        @overload
        def __get__(self, obj: Any, cls: Any) -> _T:
            ...

        def __get__(self, obj, cls):
            if obj is None:
                return self
            obj.__dict__[self.__name__] = result = self.fget(obj)
            obj._memoized_keys |= {self.__name__}
            return result

    @classmethod
    def memoized_instancemethod(cls, fn: _F) -> _F:
        """Decorate a method memoize its return value.

        :meta private:

        """

        def oneshot(self: Any, *args: Any, **kw: Any) -> Any:
            result = fn(self, *args, **kw)

            def memo(*a, **kw):
                return result
            memo.__name__ = fn.__name__
            memo.__doc__ = fn.__doc__
            self.__dict__[fn.__name__] = memo
            self._memoized_keys |= {fn.__name__}
            return result
        return update_wrapper(oneshot, fn)