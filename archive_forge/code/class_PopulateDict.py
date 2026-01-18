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
class PopulateDict(Dict[_KT, _VT]):
    """A dict which populates missing values via a creation function.

    Note the creation function takes a key, unlike
    collections.defaultdict.

    """

    def __init__(self, creator: Callable[[_KT], _VT]):
        self.creator = creator

    def __missing__(self, key: Any) -> Any:
        self[key] = val = self.creator(key)
        return val