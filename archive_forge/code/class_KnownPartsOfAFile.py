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
class KnownPartsOfAFile(BaseCache):
    """
    Cache holding known file parts.

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    data: dict
        A dictionary mapping explicit `(start, stop)` file-offset tuples
        with known bytes.
    strict: bool, default True
        Whether to fetch reads that go beyond a known byte-range boundary.
        If `False`, any read that ends outside a known part will be zero
        padded. Note that zero padding will not be used for reads that
        begin outside a known byte-range.
    """
    name: ClassVar[str] = 'parts'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, data: dict[tuple[int, int], bytes]={}, strict: bool=True, **_: Any):
        super().__init__(blocksize, fetcher, size)
        self.strict = strict
        if data:
            old_offsets = sorted(data.keys())
            offsets = [old_offsets[0]]
            blocks = [data.pop(old_offsets[0])]
            for start, stop in old_offsets[1:]:
                start0, stop0 = offsets[-1]
                if start == stop0:
                    offsets[-1] = (start0, stop)
                    blocks[-1] += data.pop((start, stop))
                else:
                    offsets.append((start, stop))
                    blocks.append(data.pop((start, stop)))
            self.data = dict(zip(offsets, blocks))
        else:
            self.data = data

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        out = b''
        for (loc0, loc1), data in self.data.items():
            if loc0 <= start < loc1:
                off = start - loc0
                out = data[off:off + stop - start]
                if not self.strict or loc0 <= stop <= loc1:
                    out += b'\x00' * (stop - start - len(out))
                    return out
                else:
                    start = loc1
                    break
        if self.fetcher is None:
            raise ValueError(f'Read is outside the known file parts: {(start, stop)}. ')
        warnings.warn(f'Read is outside the known file parts: {(start, stop)}. IO/caching performance may be poor!')
        logger.debug(f'KnownPartsOfAFile cache fetching {start}-{stop}')
        return out + super()._fetch(start, stop)