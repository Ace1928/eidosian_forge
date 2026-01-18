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
class BaseRunner(ABC):
    __slots__ = ('shutdown_callback', '_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')

    def __init__(self, *, handle_signals: bool=False, shutdown_timeout: float=60.0, **kwargs: Any) -> None:
        self.shutdown_callback: Optional[Callable[[], Awaitable[None]]] = None
        self._handle_signals = handle_signals
        self._kwargs = kwargs
        self._server: Optional[Server] = None
        self._sites: List[BaseSite] = []
        self._shutdown_timeout = shutdown_timeout

    @property
    def server(self) -> Optional[Server]:
        return self._server

    @property
    def addresses(self) -> List[Any]:
        ret: List[Any] = []
        for site in self._sites:
            server = site._server
            if server is not None:
                sockets = server.sockets
                if sockets is not None:
                    for sock in sockets:
                        ret.append(sock.getsockname())
        return ret

    @property
    def sites(self) -> Set[BaseSite]:
        return set(self._sites)

    async def setup(self) -> None:
        loop = asyncio.get_event_loop()
        if self._handle_signals:
            try:
                loop.add_signal_handler(signal.SIGINT, _raise_graceful_exit)
                loop.add_signal_handler(signal.SIGTERM, _raise_graceful_exit)
            except NotImplementedError:
                pass
        self._server = await self._make_server()

    @abstractmethod
    async def shutdown(self) -> None:
        """Call any shutdown hooks to help server close gracefully."""

    async def cleanup(self) -> None:
        for site in list(self._sites):
            await site.stop()
        if self._server:
            self._server.pre_shutdown()
            await self.shutdown()
            if self.shutdown_callback:
                await self.shutdown_callback()
            await self._server.shutdown(self._shutdown_timeout)
        await self._cleanup_server()
        self._server = None
        if self._handle_signals:
            loop = asyncio.get_running_loop()
            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            except NotImplementedError:
                pass

    @abstractmethod
    async def _make_server(self) -> Server:
        pass

    @abstractmethod
    async def _cleanup_server(self) -> None:
        pass

    def _reg_site(self, site: BaseSite) -> None:
        if site in self._sites:
            raise RuntimeError(f'Site {site} is already registered in runner {self}')
        self._sites.append(site)

    def _check_site(self, site: BaseSite) -> None:
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')

    def _unreg_site(self, site: BaseSite) -> None:
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')
        self._sites.remove(site)