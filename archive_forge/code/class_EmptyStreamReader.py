import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
class EmptyStreamReader(StreamReader):

    def __init__(self) -> None:
        self._read_eof_chunk = False

    def __repr__(self) -> str:
        return '<%s>' % self.__class__.__name__

    def exception(self) -> Optional[BaseException]:
        return None

    def set_exception(self, exc: BaseException) -> None:
        pass

    def on_eof(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            internal_logger.exception('Exception in eof callback')

    def feed_eof(self) -> None:
        pass

    def is_eof(self) -> bool:
        return True

    def at_eof(self) -> bool:
        return True

    async def wait_eof(self) -> None:
        return

    def feed_data(self, data: bytes, n: int=0) -> None:
        pass

    async def readline(self) -> bytes:
        return b''

    async def read(self, n: int=-1) -> bytes:
        return b''

    async def readany(self) -> bytes:
        return b''

    async def readchunk(self) -> Tuple[bytes, bool]:
        if not self._read_eof_chunk:
            self._read_eof_chunk = True
            return (b'', False)
        return (b'', True)

    async def readexactly(self, n: int) -> bytes:
        raise asyncio.IncompleteReadError(b'', n)

    def read_nowait(self, n: int=-1) -> bytes:
        return b''