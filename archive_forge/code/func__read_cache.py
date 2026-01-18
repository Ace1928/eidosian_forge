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