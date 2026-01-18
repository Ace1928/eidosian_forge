from __future__ import annotations
import heapq
import math
from functools import reduce
from itertools import chain, count, product
from operator import add, itemgetter, mul
from warnings import warn
import numpy as np
import tlz as toolz
from tlz import accumulate
from dask import config
from dask.array.chunk import getitem
from dask.array.core import Array, concatenate3, normalize_chunks
from dask.array.utils import validate_axis
from dask.array.wrap import empty
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
def _balance_chunksizes(chunks: tuple[int, ...]) -> tuple[int, ...]:
    """
    Balance the chunk sizes

    Parameters
    ----------
    chunks : tuple[int, ...]
        Chunk sizes for Dask array.

    Returns
    -------
    new_chunks : tuple[int, ...]
        New chunks for Dask array with balanced sizes.
    """
    median_len = np.median(chunks).astype(int)
    n_chunks = len(chunks)
    eps = median_len // 2
    if min(chunks) <= 0.5 * max(chunks):
        n_chunks -= 1
    new_chunks = [_get_chunks(sum(chunks), chunk_len) for chunk_len in range(median_len - eps, median_len + eps + 1)]
    possible_chunks = [c for c in new_chunks if len(c) == n_chunks]
    if not len(possible_chunks):
        warn('chunk size balancing not possible with given chunks. Try increasing the chunk size.')
        return chunks
    diffs = [max(c) - min(c) for c in possible_chunks]
    best_chunk_size = np.argmin(diffs)
    return possible_chunks[best_chunk_size]