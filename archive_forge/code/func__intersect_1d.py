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
def _intersect_1d(breaks):
    """
    Internal utility to intersect chunks for 1d after preprocessing.

    >>> new = cumdims_label(((2, 3), (2, 2, 1)), 'n')
    >>> old = cumdims_label(((2, 2, 1), (5,)), 'o')

    >>> _intersect_1d(_breakpoints(old[0], new[0]))  # doctest: +NORMALIZE_WHITESPACE
    [[(0, slice(0, 2, None))],
     [(1, slice(0, 2, None)), (2, slice(0, 1, None))]]
    >>> _intersect_1d(_breakpoints(old[1], new[1]))  # doctest: +NORMALIZE_WHITESPACE
    [[(0, slice(0, 2, None))],
     [(0, slice(2, 4, None))],
     [(0, slice(4, 5, None))]]

    Parameters
    ----------

    breaks: list of tuples
        Each tuple is ('o', 8) or ('n', 8)
        These are pairs of 'o' old or new 'n'
        indicator with a corresponding cumulative sum,
        or breakpoint (a position along the chunking axis).
        The list of pairs is already ordered by breakpoint.
        Note that an 'o' pair always occurs BEFORE
        an 'n' pair if both share the same breakpoint.
    Uses 'o' and 'n' to make new tuples of slices for
    the new block crosswalk to old blocks.
    """
    o_pairs = [pair for pair in breaks if pair[0] == 'o']
    last_old_chunk_idx = len(o_pairs) - 2
    last_o_br = o_pairs[-1][1]
    start = 0
    last_end = 0
    old_idx = 0
    last_o_end = 0
    ret = []
    ret_next = []
    for idx in range(1, len(breaks)):
        label, br = breaks[idx]
        last_label, last_br = breaks[idx - 1]
        if last_label == 'n':
            start = last_end
            if ret_next:
                ret.append(ret_next)
                ret_next = []
        else:
            start = 0
        end = br - last_br + start
        last_end = end
        if br == last_br:
            if label == 'o':
                old_idx += 1
                last_o_end = end
            if label == 'n' and last_label == 'n':
                if br == last_o_br:
                    slc = slice(last_o_end, last_o_end)
                    ret_next.append((last_old_chunk_idx, slc))
                    continue
            else:
                continue
        ret_next.append((old_idx, slice(start, end)))
        if label == 'o':
            old_idx += 1
            start = 0
            last_o_end = end
    if ret_next:
        ret.append(ret_next)
    return ret