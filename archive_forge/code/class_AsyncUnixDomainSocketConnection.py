import uuid
import logging
import asyncio
import copy
import enum
import errno
import inspect
import io
import os
import socket
import ssl
import threading
import weakref
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
import async_timeout
from aiokeydb.v1.backoff import NoBackoff
from aiokeydb.v1.asyncio.retry import Retry
from aiokeydb.v1.compat import Protocol, TypedDict
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from aiokeydb.v1.typing import EncodableT, EncodedT
from aiokeydb.v1.utils import HIREDIS_AVAILABLE, str_if_bytes, set_ulimits
class AsyncUnixDomainSocketConnection(AsyncConnection):

    def __init__(self, *, path: str='', db: Union[str, int]=0, username: Optional[str]=None, password: Optional[str]=None, socket_timeout: Optional[float]=None, socket_connect_timeout: Optional[float]=None, encoding: str='utf-8', encoding_errors: str='strict', decode_responses: bool=False, retry_on_timeout: bool=False, retry_on_error: Union[list, _Sentinel]=SENTINEL, parser_class: Type[BaseParser]=DefaultParser, socket_read_size: int=65536, health_check_interval: float=0.0, client_name: str=None, retry: Optional[Retry]=None, keydb_connect_func=None, credential_provider: Optional[CredentialProvider]=None):
        """
        Initialize a new UnixDomainSocketConnection.
        To specify a retry policy, first set `retry_on_timeout` to `True`
        then set `retry` to a valid `Retry` object
        """
        if (username or password) and credential_provider is not None:
            raise DataError("'username' and 'password' cannot be passed along with 'credential_provider'. Please provide only one of the following arguments: \n1. 'password' and (optional) 'username'\n2. 'credential_provider'")
        self.pid = os.getpid()
        self.path = path
        self.db = db
        self.username = username
        self.password = password
        self.credential_provider = credential_provider
        self.client_name = client_name
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout or socket_timeout or None
        self.retry_on_timeout = retry_on_timeout
        if retry_on_error is SENTINEL:
            retry_on_error = []
        if retry_on_timeout:
            retry_on_error.append(TimeoutError)
        self.retry_on_error = retry_on_error
        if retry_on_error:
            if retry is None:
                self.retry = Retry(NoBackoff(), 1)
            else:
                self.retry = copy.deepcopy(retry)
            self.retry.update_supported_errors(retry_on_error)
        else:
            self.retry = Retry(NoBackoff(), 0)
        self.health_check_interval = health_check_interval
        self.next_health_check = -1
        self.keydb_connect_func = keydb_connect_func
        self.encoder = Encoder(encoding, encoding_errors, decode_responses)
        self._sock = None
        self._reader = None
        self._writer = None
        self._socket_read_size = socket_read_size
        self.set_parser(parser_class)
        self._connect_callbacks = []
        self._buffer_cutoff = 6000
        self._lock = asyncio.Lock()

    def repr_pieces(self) -> Iterable[Tuple[str, Union[str, int]]]:
        pieces = [('path', self.path), ('db', self.db)]
        if self.client_name:
            pieces.append(('client_name', self.client_name))
        return pieces

    async def _connect(self):
        async with async_timeout.timeout(self.socket_connect_timeout):
            reader, writer = await asyncio.open_unix_connection(path=self.path)
        self._reader = reader
        self._writer = writer
        await self.on_connect()

    def _error_message(self, exception):
        if len(exception.args) == 1:
            return f'Error connecting to unix socket: {self.path}. {exception.args[0]}.'
        else:
            return f'Error {exception.args[0]} connecting to unix socket: {self.path}. {exception.args[1]}.'