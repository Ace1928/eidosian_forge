import asyncio
import signal
import socket
import warnings
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, Set
from yarl import URL
from .typedefs import PathLike
from .web_app import Application
from .web_server import Server
class UnixSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(self, runner: 'BaseRunner', path: PathLike, *, shutdown_timeout: float=60.0, ssl_context: Optional[SSLContext]=None, backlog: int=128) -> None:
        super().__init__(runner, shutdown_timeout=shutdown_timeout, ssl_context=ssl_context, backlog=backlog)
        self._path = path

    @property
    def name(self) -> str:
        scheme = 'https' if self._ssl_context else 'http'
        return f'{scheme}://unix:{self._path}:'

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_unix_server(server, self._path, ssl=self._ssl_context, backlog=self._backlog)