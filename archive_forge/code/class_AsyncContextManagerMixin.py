from __future__ import annotations
import math
from typing import TYPE_CHECKING, Protocol
import attrs
import trio
from . import _core
from ._core import Abort, ParkingLot, RaiseCancelT, enable_ki_protection
from ._util import final
class AsyncContextManagerMixin:

    @enable_ki_protection
    async def __aenter__(self: _HasAcquireRelease) -> None:
        await self.acquire()

    @enable_ki_protection
    async def __aexit__(self: _HasAcquireRelease, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.release()