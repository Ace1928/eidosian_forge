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
class _AsyncSessionContextManager(Generic[_AS]):
    __slots__ = ('async_session', 'trans')
    async_session: _AS
    trans: AsyncSessionTransaction

    def __init__(self, async_session: _AS):
        self.async_session = async_session

    async def __aenter__(self) -> _AS:
        self.trans = self.async_session.begin()
        await self.trans.__aenter__()
        return self.async_session

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:

        async def go() -> None:
            await self.trans.__aexit__(type_, value, traceback)
            await self.async_session.__aexit__(type_, value, traceback)
        task = asyncio.create_task(go())
        await asyncio.shield(task)