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
class BlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """
    name = 'blockcache'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, maxblocks: int=32) -> None:
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = functools.lru_cache(maxblocks)(self._fetch_block)

    def __repr__(self) -> str:
        return f'<BlockCache blocksize={self.blocksize}, size={self.size}, nblocks={self.nblocks}>'

    def cache_info(self):
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        del state['_fetch_block_cached']
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fetch_block_cached = functools.lru_cache(state['maxblocks'])(self._fetch_block)

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b''
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)
        return self._read_cache(start, end, start_block_number=start_block_number, end_block_number=end_block_number)

    def _fetch_block(self, block_number: int) -> bytes:
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(f"'block_number={block_number}' is greater than the number of blocks ({self.nblocks})")
        start = block_number * self.blocksize
        end = start + self.blocksize
        logger.info('BlockCache fetching block %d', block_number)
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(self, start: int, end: int, start_block_number: int, end_block_number: int) -> bytes:
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize
        if start_block_number == end_block_number:
            block: bytes = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]
        else:
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]
            out.extend(map(self._fetch_block_cached, range(start_block_number + 1, end_block_number)))
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])
            return b''.join(out)