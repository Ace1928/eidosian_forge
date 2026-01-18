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
class BackgroundBlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks with pre-loading of
    the next block in the background.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored. If the
    next block is not in cache, it is loaded in a separate thread
    in non-blocking way.

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
    name: ClassVar[str] = 'background'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, maxblocks: int=32) -> None:
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, maxblocks)
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number: int | None = None
        self._fetch_future: Future[bytes] | None = None
        self._fetch_future_lock = threading.Lock()

    def __repr__(self) -> str:
        return f'<BackgroundBlockCache blocksize={self.blocksize}, size={self.size}, nblocks={self.nblocks}>'

    def cache_info(self) -> UpdatableLRU.CacheInfo:
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
        del state['_thread_executor']
        del state['_fetch_future_block_number']
        del state['_fetch_future']
        del state['_fetch_future_lock']
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, state['maxblocks'])
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number = None
        self._fetch_future = None
        self._fetch_future_lock = threading.Lock()

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b''
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize
        fetch_future_block_number = None
        fetch_future = None
        with self._fetch_future_lock:
            if self._fetch_future is not None:
                assert self._fetch_future_block_number is not None
                if self._fetch_future.done():
                    logger.info('BlockCache joined background fetch without waiting.')
                    self._fetch_block_cached.add_key(self._fetch_future.result(), self._fetch_future_block_number)
                    self._fetch_future_block_number = None
                    self._fetch_future = None
                else:
                    must_join = bool(start_block_number <= self._fetch_future_block_number <= end_block_number)
                    if must_join:
                        fetch_future_block_number = self._fetch_future_block_number
                        fetch_future = self._fetch_future
                        self._fetch_future_block_number = None
                        self._fetch_future = None
        if fetch_future is not None:
            logger.info('BlockCache waiting for background fetch.')
            self._fetch_block_cached.add_key(fetch_future.result(), fetch_future_block_number)
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)
        end_block_plus_1 = end_block_number + 1
        with self._fetch_future_lock:
            if self._fetch_future is None and end_block_plus_1 <= self.nblocks and (not self._fetch_block_cached.is_key_cached(end_block_plus_1)):
                self._fetch_future_block_number = end_block_plus_1
                self._fetch_future = self._thread_executor.submit(self._fetch_block, end_block_plus_1, 'async')
        return self._read_cache(start, end, start_block_number=start_block_number, end_block_number=end_block_number)

    def _fetch_block(self, block_number: int, log_info: str='sync') -> bytes:
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(f"'block_number={block_number}' is greater than the number of blocks ({self.nblocks})")
        start = block_number * self.blocksize
        end = start + self.blocksize
        logger.info('BlockCache fetching block (%s) %d', log_info, block_number)
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
            block = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]
        else:
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]
            out.extend(map(self._fetch_block_cached, range(start_block_number + 1, end_block_number)))
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])
            return b''.join(out)