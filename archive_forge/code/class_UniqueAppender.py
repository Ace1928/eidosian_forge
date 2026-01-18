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
class UniqueAppender(Generic[_T]):
    """Appends items to a collection ensuring uniqueness.

    Additional appends() of the same object are ignored.  Membership is
    determined by identity (``is a``) not equality (``==``).
    """
    __slots__ = ('data', '_data_appender', '_unique')
    data: Union[Iterable[_T], Set[_T], List[_T]]
    _data_appender: Callable[[_T], None]
    _unique: Dict[int, Literal[True]]

    def __init__(self, data: Union[Iterable[_T], Set[_T], List[_T]], via: Optional[str]=None):
        self.data = data
        self._unique = {}
        if via:
            self._data_appender = getattr(data, via)
        elif hasattr(data, 'append'):
            self._data_appender = cast('List[_T]', data).append
        elif hasattr(data, 'add'):
            self._data_appender = cast('Set[_T]', data).add

    def append(self, item: _T) -> None:
        id_ = id(item)
        if id_ not in self._unique:
            self._data_appender(item)
            self._unique[id_] = True

    def __iter__(self) -> Iterator[_T]:
        return iter(self.data)