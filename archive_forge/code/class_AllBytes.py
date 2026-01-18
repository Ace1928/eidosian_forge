from __future__ import annotations
import collections
import functools
import logging
import math
import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
class AllBytes(BaseCache):
    """Cache entire contents of the file"""
    name: ClassVar[str] = 'all'

    def __init__(self, blocksize: int | None=None, fetcher: Fetcher | None=None, size: int | None=None, data: bytes | None=None) -> None:
        super().__init__(blocksize, fetcher, size)
        if data is None:
            data = self.fetcher(0, self.size)
        self.data = data

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        return self.data[start:stop]