from __future__ import annotations
import heapq
import math
import random as rnd
from functools import partial
from itertools import islice
from dask.bag.core import Bag
def _sample_with_replacement_map_partitions(population, k):
    """
    Reservoir sampling with replacement, the main idea is to use k reservoirs of size 1
    See Section Applications in http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
    """
    stream = iter(population)
    e = next(stream)
    reservoir, stream_length = ([e for _ in range(k)], 1)
    w = [rnd.random() for _ in range(k)]
    nxt = [_geometric(wi) for wi in w]
    min_nxt = min(nxt)
    for i, e in enumerate(stream, 1):
        if i == min_nxt:
            for j, n in enumerate(nxt):
                if n == min_nxt:
                    reservoir[j] = e
                    w[j] *= rnd.random()
                    nxt[j] += _geometric(w[j])
            min_nxt = min(nxt)
        stream_length += 1
    return (reservoir, stream_length)