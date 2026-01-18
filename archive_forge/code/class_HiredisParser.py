import asyncio
import enum
import errno
import inspect
import io
import os
import socket
import ssl
import threading
import warnings
from distutils.version import StrictVersion
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
import async_timeout
from .compat import Protocol, TypedDict
from .exceptions import (
from .utils import str_if_bytes
class HiredisParser(BaseParser):
    """Parser class for connections using Hiredis"""
    __slots__ = BaseParser.__slots__ + ('_next_response', '_reader', '_socket_timeout')
    _next_response: bool

    def __init__(self, socket_read_size: int):
        if not HIREDIS_AVAILABLE:
            raise RedisError('Hiredis is not available.')
        super().__init__(socket_read_size=socket_read_size)
        self._reader: Optional[hiredis.Reader] = None
        self._socket_timeout: Optional[float] = None

    def on_connect(self, connection: 'Connection'):
        self._stream = connection._reader
        kwargs: _HiredisReaderArgs = {'protocolError': InvalidResponse, 'replyError': self.parse_error}
        if connection.encoder.decode_responses:
            kwargs['encoding'] = connection.encoder.encoding
            kwargs['errors'] = connection.encoder.encoding_errors
        self._reader = hiredis.Reader(**kwargs)
        self._next_response = False
        self._socket_timeout = connection.socket_timeout

    def on_disconnect(self):
        self._stream = None
        self._reader = None
        self._next_response = False

    async def can_read(self, timeout: float):
        if not self._reader:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._next_response is False:
            self._next_response = self._reader.gets()
        if self._next_response is False:
            return await self.read_from_socket(timeout=timeout, raise_on_timeout=False)
        return True

    async def read_from_socket(self, timeout: Union[float, None, _Sentinel]=SENTINEL, raise_on_timeout: bool=True):
        if self._stream is None or self._reader is None:
            raise RedisError('Parser already closed.')
        timeout = self._socket_timeout if timeout is SENTINEL else timeout
        try:
            async with async_timeout.timeout(timeout):
                buffer = await self._stream.read(self._read_size)
            if not isinstance(buffer, bytes) or len(buffer) == 0:
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from None
            self._reader.feed(buffer)
            return True
        except asyncio.CancelledError:
            raise
        except (socket.timeout, asyncio.TimeoutError):
            if raise_on_timeout:
                raise TimeoutError('Timeout reading from socket') from None
            return False
        except NONBLOCKING_EXCEPTIONS as ex:
            allowed = NONBLOCKING_EXCEPTION_ERROR_NUMBERS.get(ex.__class__, -1)
            if not raise_on_timeout and ex.errno == allowed:
                return False
            raise ConnectionError(f'Error while reading from socket: {ex.args}')

    async def read_response(self) -> Union[EncodableT, List[EncodableT]]:
        if not self._stream or not self._reader:
            self.on_disconnect()
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from None
        response: Union[EncodableT, ConnectionError, List[Union[EncodableT, ConnectionError]]]
        if self._next_response is not False:
            response = self._next_response
            self._next_response = False
            return response
        response = self._reader.gets()
        while response is False:
            await self.read_from_socket()
            response = self._reader.gets()
        if isinstance(response, ConnectionError):
            raise response
        elif isinstance(response, list) and response and isinstance(response[0], ConnectionError):
            raise response[0]
        return cast(Union[EncodableT, List[EncodableT]], response)