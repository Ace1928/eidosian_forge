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