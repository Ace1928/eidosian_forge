import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
class NamedPipeConnector(BaseConnector):
    """Named pipe connector.

    Only supported by the proactor event loop.
    See also: https://docs.python.org/3/library/asyncio-eventloop.html

    path - Windows named pipe path.
    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """

    def __init__(self, path: str, force_close: bool=False, keepalive_timeout: Union[object, float, None]=sentinel, limit: int=100, limit_per_host: int=0, loop: Optional[asyncio.AbstractEventLoop]=None) -> None:
        super().__init__(force_close=force_close, keepalive_timeout=keepalive_timeout, limit=limit, limit_per_host=limit_per_host, loop=loop)
        if not isinstance(self._loop, asyncio.ProactorEventLoop):
            raise RuntimeError('Named Pipes only available in proactor loop under windows')
        self._path = path

    @property
    def path(self) -> str:
        """Path to the named pipe."""
        return self._path

    async def _create_connection(self, req: ClientRequest, traces: List['Trace'], timeout: 'ClientTimeout') -> ResponseHandler:
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                _, proto = await self._loop.create_pipe_connection(self._factory, self._path)
                await asyncio.sleep(0)
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise ClientConnectorError(req.connection_key, exc) from exc
        return cast(ResponseHandler, proto)