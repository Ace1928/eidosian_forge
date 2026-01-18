from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
class _LockstepReceiveStream(ReceiveStream):

    def __init__(self, lbq: _LockstepByteQueue):
        self._lbq = lbq

    def close(self) -> None:
        self._lbq.close_receiver()

    async def aclose(self) -> None:
        self.close()
        await _core.checkpoint()

    async def receive_some(self, max_bytes: int | None=None) -> bytes | bytearray:
        return await self._lbq.receive_some(max_bytes)