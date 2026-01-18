from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def _read_iter(self, nbytes: int):
    assert nbytes <= self.bytes_buffered
    while True:
        chunk = self.chunks.popleft()
        self.bytes_buffered -= len(chunk)
        if nbytes <= len(chunk):
            break
        nbytes -= len(chunk)
        yield chunk
    chunk, rem = (chunk[:nbytes], chunk[nbytes:])
    if rem:
        self.chunks.appendleft(rem)
        self.bytes_buffered += len(rem)
    yield chunk