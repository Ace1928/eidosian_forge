from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
class _UnboundedByteQueue:

    def __init__(self) -> None:
        self._data = bytearray()
        self._closed = False
        self._lot = _core.ParkingLot()
        self._fetch_lock = _util.ConflictDetector('another task is already fetching data')

    def close(self) -> None:
        self._closed = True
        self._lot.unpark_all()

    def close_and_wipe(self) -> None:
        self._data = bytearray()
        self.close()

    def put(self, data: bytes | bytearray | memoryview) -> None:
        if self._closed:
            raise _core.ClosedResourceError('virtual connection closed')
        self._data += data
        self._lot.unpark_all()

    def _check_max_bytes(self, max_bytes: int | None) -> None:
        if max_bytes is None:
            return
        max_bytes = operator.index(max_bytes)
        if max_bytes < 1:
            raise ValueError('max_bytes must be >= 1')

    def _get_impl(self, max_bytes: int | None) -> bytearray:
        assert self._closed or self._data
        if max_bytes is None:
            max_bytes = len(self._data)
        if self._data:
            chunk = self._data[:max_bytes]
            del self._data[:max_bytes]
            assert chunk
            return chunk
        else:
            return bytearray()

    def get_nowait(self, max_bytes: int | None=None) -> bytearray:
        with self._fetch_lock:
            self._check_max_bytes(max_bytes)
            if not self._closed and (not self._data):
                raise _core.WouldBlock
            return self._get_impl(max_bytes)

    async def get(self, max_bytes: int | None=None) -> bytearray:
        with self._fetch_lock:
            self._check_max_bytes(max_bytes)
            if not self._closed and (not self._data):
                await self._lot.park()
            else:
                await _core.checkpoint()
            return self._get_impl(max_bytes)