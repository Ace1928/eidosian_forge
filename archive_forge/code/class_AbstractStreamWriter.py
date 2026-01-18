import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sized
from http.cookies import BaseCookie, Morsel
from typing import (
from multidict import CIMultiDict
from yarl import URL
from .helpers import get_running_loop
from .typedefs import LooseCookies
class AbstractStreamWriter(ABC):
    """Abstract stream writer."""
    buffer_size = 0
    output_size = 0
    length: Optional[int] = 0

    @abstractmethod
    async def write(self, chunk: bytes) -> None:
        """Write chunk into stream."""

    @abstractmethod
    async def write_eof(self, chunk: bytes=b'') -> None:
        """Write last chunk."""

    @abstractmethod
    async def drain(self) -> None:
        """Flush the write buffer."""

    @abstractmethod
    def enable_compression(self, encoding: str='deflate') -> None:
        """Enable HTTP body compression"""

    @abstractmethod
    def enable_chunking(self) -> None:
        """Enable HTTP chunked mode"""

    @abstractmethod
    async def write_headers(self, status_line: str, headers: 'CIMultiDict[str]') -> None:
        """Write HTTP headers"""