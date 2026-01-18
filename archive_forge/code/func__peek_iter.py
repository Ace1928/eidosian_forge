from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def _peek_iter(self, nbytes: int):
    assert nbytes <= self.bytes_buffered
    for chunk in self.chunks:
        chunk = chunk[:nbytes]
        nbytes -= len(chunk)
        yield chunk
        if nbytes <= 0:
            break