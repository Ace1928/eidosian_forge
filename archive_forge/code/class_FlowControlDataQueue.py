import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
class FlowControlDataQueue(DataQueue[_T]):
    """FlowControlDataQueue resumes and pauses an underlying stream.

    It is a destination for parsed data.
    """

    def __init__(self, protocol: BaseProtocol, limit: int, *, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(loop=loop)
        self._protocol = protocol
        self._limit = limit * 2

    def feed_data(self, data: _T, size: int=0) -> None:
        super().feed_data(data, size)
        if self._size > self._limit and (not self._protocol._reading_paused):
            self._protocol.pause_reading()

    async def read(self) -> _T:
        try:
            return await super().read()
        finally:
            if self._size < self._limit and self._protocol._reading_paused:
                self._protocol.resume_reading()