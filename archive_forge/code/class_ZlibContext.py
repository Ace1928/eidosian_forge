from __future__ import annotations
import warnings
from typing import Any, Iterable, Optional, Union
from pymongo.hello import HelloCompat
from pymongo.monitoring import _SENSITIVE_COMMANDS
class ZlibContext:
    compressor_id = 2

    def __init__(self, level: int):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, self.level)