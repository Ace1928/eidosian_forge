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
class _JoinedListener(_CompoundListener[_ET]):
    __slots__ = ('parent_dispatch', 'name', 'local', 'parent_listeners')
    parent_dispatch: _DispatchCommon[_ET]
    name: str
    local: _InstanceLevelDispatch[_ET]
    parent_listeners: Collection[_ListenerFnType]

    def __init__(self, parent_dispatch: _DispatchCommon[_ET], name: str, local: _EmptyListener[_ET]):
        self._exec_once = False
        self.parent_dispatch = parent_dispatch
        self.name = name
        self.local = local
        self.parent_listeners = self.local
    if not typing.TYPE_CHECKING:

        @property
        def listeners(self) -> Collection[_ListenerFnType]:
            return getattr(self.parent_dispatch, self.name)

    def _adjust_fn_spec(self, fn: _ListenerFnType, named: bool) -> _ListenerFnType:
        return self.local._adjust_fn_spec(fn, named)

    def for_modify(self, obj: _DispatchCommon[_ET]) -> _JoinedListener[_ET]:
        self.local = self.parent_listeners = self.local.for_modify(obj)
        return self

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self.local.insert(event_key, propagate)

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self.local.append(event_key, propagate)

    def remove(self, event_key: _EventKey[_ET]) -> None:
        self.local.remove(event_key)

    def clear(self) -> None:
        raise NotImplementedError()