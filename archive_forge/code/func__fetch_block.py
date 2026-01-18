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