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
class _InstanceLevelDispatch(RefCollection[_ET], Collection[_ListenerFnType]):
    __slots__ = ()
    parent: _ClsLevelDispatch[_ET]

    def _adjust_fn_spec(self, fn: _ListenerFnType, named: bool) -> _ListenerFnType:
        return self.parent._adjust_fn_spec(fn, named)

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[_ListenerFnType]:
        raise NotImplementedError()

    def __bool__(self) -> bool:
        raise NotImplementedError()

    def exec_once(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def exec_once_unless_exception(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def _exec_w_sync_on_first_run(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        raise NotImplementedError()

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        raise NotImplementedError()

    def remove(self, event_key: _EventKey[_ET]) -> None:
        raise NotImplementedError()

    def for_modify(self, obj: _DispatchCommon[_ET]) -> _InstanceLevelDispatch[_ET]:
        """Return an event collection which can be modified.

        For _ClsLevelDispatch at the class level of
        a dispatcher, this returns self.

        """
        return self