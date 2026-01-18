import json
import mmap
import os
import struct
from typing import List
def _init_value(self, key):
    """Initialize a value. Lock must be held by caller."""
    encoded = key.encode('utf-8')
    padded = encoded + b' ' * (8 - (len(encoded) + 4) % 8)
    value = struct.pack(f'i{len(padded)}sdd'.encode(), len(encoded), padded, 0.0, 0.0)
    while self._used + len(value) > self._capacity:
        self._capacity *= 2
        self._f.truncate(self._capacity)
        self._m = mmap.mmap(self._f.fileno(), self._capacity)
    self._m[self._used:self._used + len(value)] = value
    self._used += len(value)
    _pack_integer(self._m, 0, self._used)
    self._positions[key] = self._used - 16