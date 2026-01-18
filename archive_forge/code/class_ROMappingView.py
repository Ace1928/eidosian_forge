from __future__ import annotations
from abc import ABC
import collections.abc as collections_abc
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..sql import util as sql_util
from ..util import deprecated
from ..util._has_cy import HAS_CYEXTENSION
class ROMappingView(ABC):
    __slots__ = ()
    _items: Sequence[Any]
    _mapping: Mapping['_KeyType', Any]

    def __init__(self, mapping: Mapping['_KeyType', Any], items: Sequence[Any]):
        self._mapping = mapping
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return '{0.__class__.__name__}({0._mapping!r})'.format(self)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __contains__(self, item: Any) -> bool:
        return item in self._items

    def __eq__(self, other: Any) -> bool:
        return list(other) == list(self)

    def __ne__(self, other: Any) -> bool:
        return list(other) != list(self)