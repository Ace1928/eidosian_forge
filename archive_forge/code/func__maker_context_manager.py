from __future__ import annotations
import asyncio
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import engine
from .base import ReversibleProxy
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import util
from ...orm import close_all_sessions as _sync_close_all_sessions
from ...orm import object_session
from ...orm import Session
from ...orm import SessionTransaction
from ...orm import state as _instance_state
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
def _maker_context_manager(self: _AS) -> _AsyncSessionContextManager[_AS]:
    return _AsyncSessionContextManager(self)