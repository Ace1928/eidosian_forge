from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
class _LockstepByteQueue:

    def __init__(self) -> None:
        self._data = bytearray()
        self._sender_closed = False
        self._receiver_closed = False
        self._receiver_waiting = False
        self._waiters = _core.ParkingLot()
        self._send_conflict_detector = _util.ConflictDetector('another task is already sending')
        self._receive_conflict_detector = _util.ConflictDetector('another task is already receiving')

    def _something_happened(self) -> None:
        self._waiters.unpark_all()

    async def _wait_for(self, fn: Callable[[], bool]) -> None:
        while True:
            if fn():
                break
            if self._sender_closed or self._receiver_closed:
                break
            await self._waiters.park()
        await _core.checkpoint()

    def close_sender(self) -> None:
        self._sender_closed = True
        self._something_happened()

    def close_receiver(self) -> None:
        self._receiver_closed = True
        self._something_happened()

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        with self._send_conflict_detector:
            if self._sender_closed:
                raise _core.ClosedResourceError
            if self._receiver_closed:
                raise _core.BrokenResourceError
            assert not self._data
            self._data += data
            self._something_happened()
            await self._wait_for(lambda: self._data == b'')
            if self._sender_closed:
                raise _core.ClosedResourceError
            if self._data and self._receiver_closed:
                raise _core.BrokenResourceError

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_conflict_detector:
            if self._sender_closed:
                raise _core.ClosedResourceError
            if self._receiver_closed:
                await _core.checkpoint()
                return
            await self._wait_for(lambda: self._receiver_waiting)
            if self._sender_closed:
                raise _core.ClosedResourceError

    async def receive_some(self, max_bytes: int | None=None) -> bytes | bytearray:
        with self._receive_conflict_detector:
            if max_bytes is not None:
                max_bytes = operator.index(max_bytes)
                if max_bytes < 1:
                    raise ValueError('max_bytes must be >= 1')
            if self._receiver_closed:
                raise _core.ClosedResourceError
            self._receiver_waiting = True
            self._something_happened()
            try:
                await self._wait_for(lambda: self._data != b'')
            finally:
                self._receiver_waiting = False
            if self._receiver_closed:
                raise _core.ClosedResourceError
            if self._data:
                got = self._data[:max_bytes]
                del self._data[:max_bytes]
                self._something_happened()
                return got
            else:
                assert self._sender_closed
                return b''