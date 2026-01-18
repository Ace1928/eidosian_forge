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
def _check_site(self, site: BaseSite) -> None:
    if site not in self._sites:
        raise RuntimeError(f'Site {site} is not registered in runner {self}')