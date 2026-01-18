import sys
from abc import ABC
from asyncio import IncompleteReadError, StreamReader, TimeoutError
from typing import List, Optional, Union
from ..exceptions import (
from ..typing import EncodableT
from .encoders import Encoder
from .socket import SERVER_CLOSED_CONNECTION_ERROR, SocketBuffer
class _AsyncRESPBase(AsyncBaseParser):
    """Base class for async resp parsing"""
    __slots__ = AsyncBaseParser.__slots__ + ('encoder', '_buffer', '_pos', '_chunks')

    def __init__(self, socket_read_size: int):
        super().__init__(socket_read_size)
        self.encoder: Optional[Encoder] = None
        self._buffer = b''
        self._chunks = []
        self._pos = 0

    def _clear(self):
        self._buffer = b''
        self._chunks.clear()

    def on_connect(self, connection):
        """Called when the stream connects"""
        self._stream = connection._reader
        if self._stream is None:
            raise RedisError('Buffer is closed.')
        self.encoder = connection.encoder
        self._clear()
        self._connected = True

    def on_disconnect(self):
        """Called when the stream disconnects"""
        self._connected = False

    async def can_read_destructive(self) -> bool:
        if not self._connected:
            raise RedisError('Buffer is closed.')
        if self._buffer:
            return True
        try:
            async with async_timeout(0):
                return await self._stream.read(1)
        except TimeoutError:
            return False

    async def _read(self, length: int) -> bytes:
        """
        Read `length` bytes of data.  These are assumed to be followed
        by a '\r
' terminator which is subsequently discarded.
        """
        want = length + 2
        end = self._pos + want
        if len(self._buffer) >= end:
            result = self._buffer[self._pos:end - 2]
        else:
            tail = self._buffer[self._pos:]
            try:
                data = await self._stream.readexactly(want - len(tail))
            except IncompleteReadError as error:
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from error
            result = (tail + data)[:-2]
            self._chunks.append(data)
        self._pos += want
        return result

    async def _readline(self) -> bytes:
        """
        read an unknown number of bytes up to the next '\r
'
        line separator, which is discarded.
        """
        found = self._buffer.find(b'\r\n', self._pos)
        if found >= 0:
            result = self._buffer[self._pos:found]
        else:
            tail = self._buffer[self._pos:]
            data = await self._stream.readline()
            if not data.endswith(b'\r\n'):
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
            result = (tail + data)[:-2]
            self._chunks.append(data)
        self._pos += len(result) + 2
        return result