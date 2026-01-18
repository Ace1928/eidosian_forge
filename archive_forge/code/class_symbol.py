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
class symbol(int):
    """A constant symbol.

    >>> symbol('foo') is symbol('foo')
    True
    >>> symbol('foo')
    <symbol 'foo>

    A slight refinement of the MAGICCOOKIE=object() pattern.  The primary
    advantage of symbol() is its repr().  They are also singletons.

    Repeated calls of symbol('name') will all return the same instance.

    """
    name: str
    symbols: Dict[str, symbol] = {}
    _lock = threading.Lock()

    def __new__(cls, name: str, doc: Optional[str]=None, canonical: Optional[int]=None) -> symbol:
        with cls._lock:
            sym = cls.symbols.get(name)
            if sym is None:
                assert isinstance(name, str)
                if canonical is None:
                    canonical = hash(name)
                sym = int.__new__(symbol, canonical)
                sym.name = name
                if doc:
                    sym.__doc__ = doc
                cls.symbols[name] = sym
            elif canonical and canonical != sym:
                raise TypeError(f"Can't replace canonical symbol for {name!r} with new int value {canonical}")
            return sym

    def __reduce__(self):
        return (symbol, (self.name, 'x', int(self)))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'symbol({self.name!r})'