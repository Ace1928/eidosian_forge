from __future__ import annotations
import collections
from itertools import chain
import threading
from types import TracebackType
import typing
from typing import Any
from typing import cast
from typing import Collection
from typing import Deque
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import legacy
from . import registry
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
from ..util.concurrency import AsyncAdaptedLock
from ..util.typing import Protocol
class _empty_collection(Collection[_T]):

    def append(self, element: _T) -> None:
        pass

    def appendleft(self, element: _T) -> None:
        pass

    def extend(self, other: Sequence[_T]) -> None:
        pass

    def remove(self, element: _T) -> None:
        pass

    def __contains__(self, element: Any) -> bool:
        return False

    def __iter__(self) -> Iterator[_T]:
        return iter([])

    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        return 0