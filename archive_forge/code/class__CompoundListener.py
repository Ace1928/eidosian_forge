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
class _CompoundListener(_InstanceLevelDispatch[_ET]):
    __slots__ = ('_exec_once_mutex', '_exec_once', '_exec_w_sync_once', '_is_asyncio')
    _exec_once_mutex: _MutexProtocol
    parent_listeners: Collection[_ListenerFnType]
    listeners: Collection[_ListenerFnType]
    _exec_once: bool
    _exec_w_sync_once: bool

    def __init__(self, *arg: Any, **kw: Any):
        super().__init__(*arg, **kw)
        self._is_asyncio = False

    def _set_asyncio(self) -> None:
        self._is_asyncio = True

    def _memoized_attr__exec_once_mutex(self) -> _MutexProtocol:
        if self._is_asyncio:
            return AsyncAdaptedLock()
        else:
            return threading.Lock()

    def _exec_once_impl(self, retry_on_exception: bool, *args: Any, **kw: Any) -> None:
        with self._exec_once_mutex:
            if not self._exec_once:
                try:
                    self(*args, **kw)
                    exception = False
                except:
                    exception = True
                    raise
                finally:
                    if not exception or not retry_on_exception:
                        self._exec_once = True

    def exec_once(self, *args: Any, **kw: Any) -> None:
        """Execute this event, but only if it has not been
        executed already for this collection."""
        if not self._exec_once:
            self._exec_once_impl(False, *args, **kw)

    def exec_once_unless_exception(self, *args: Any, **kw: Any) -> None:
        """Execute this event, but only if it has not been
        executed already for this collection, or was called
        by a previous exec_once_unless_exception call and
        raised an exception.

        If exec_once was already called, then this method will never run
        the callable regardless of whether it raised or not.

        .. versionadded:: 1.3.8

        """
        if not self._exec_once:
            self._exec_once_impl(True, *args, **kw)

    def _exec_w_sync_on_first_run(self, *args: Any, **kw: Any) -> None:
        """Execute this event, and use a mutex if it has not been
        executed already for this collection, or was called
        by a previous _exec_w_sync_on_first_run call and
        raised an exception.

        If _exec_w_sync_on_first_run was already called and didn't raise an
        exception, then a mutex is not used.

        .. versionadded:: 1.4.11

        """
        if not self._exec_w_sync_once:
            with self._exec_once_mutex:
                try:
                    self(*args, **kw)
                except:
                    raise
                else:
                    self._exec_w_sync_once = True
        else:
            self(*args, **kw)

    def __call__(self, *args: Any, **kw: Any) -> None:
        """Execute this event."""
        for fn in self.parent_listeners:
            fn(*args, **kw)
        for fn in self.listeners:
            fn(*args, **kw)

    def __contains__(self, item: Any) -> bool:
        return item in self.parent_listeners or item in self.listeners

    def __len__(self) -> int:
        return len(self.parent_listeners) + len(self.listeners)

    def __iter__(self) -> Iterator[_ListenerFnType]:
        return chain(self.parent_listeners, self.listeners)

    def __bool__(self) -> bool:
        return bool(self.listeners or self.parent_listeners)