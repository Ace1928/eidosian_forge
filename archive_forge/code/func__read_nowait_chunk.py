import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
def _read_nowait_chunk(self, n: int) -> bytes:
    first_buffer = self._buffer[0]
    offset = self._buffer_offset
    if n != -1 and len(first_buffer) - offset > n:
        data = first_buffer[offset:offset + n]
        self._buffer_offset += n
    elif offset:
        self._buffer.popleft()
        data = first_buffer[offset:]
        self._buffer_offset = 0
    else:
        data = self._buffer.popleft()
    self._size -= len(data)
    self._cursor += len(data)
    chunk_splits = self._http_chunk_splits
    while chunk_splits and chunk_splits[0] < self._cursor:
        chunk_splits.pop(0)
    if self._size < self._low_water and self._protocol._reading_paused:
        self._protocol.resume_reading()
    return data