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
class ReadAheadCache(BaseCache):
    """Cache which reads only when we get beyond a block of data

    This is a much simpler version of BytesCache, and does not attempt to
    fill holes in the cache or keep fragments alive. It is best suited to
    many small reads in a sequential order (e.g., reading lines from a file).
    """
    name = 'readahead'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache = b''
        self.start = 0
        self.end = 0

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size or start >= end:
            return b''
        l = end - start
        if start >= self.start and end <= self.end:
            return self.cache[start - self.start:end - self.start]
        elif self.start <= start < self.end:
            part = self.cache[start - self.start:]
            l -= len(part)
            start = self.end
        else:
            part = b''
        end = min(self.size, end + self.blocksize)
        self.cache = self.fetcher(start, end)
        self.start = start
        self.end = self.start + len(self.cache)
        return part + self.cache[:l]