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