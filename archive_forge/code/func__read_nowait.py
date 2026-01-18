import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
def _read_nowait(self, n: int) -> bytes:
    """Read not more than n bytes, or whole buffer if n == -1"""
    self._timer.assert_timeout()
    chunks = []
    while self._buffer:
        chunk = self._read_nowait_chunk(n)
        chunks.append(chunk)
        if n != -1:
            n -= len(chunk)
            if n == 0:
                break
    return b''.join(chunks) if chunks else b''