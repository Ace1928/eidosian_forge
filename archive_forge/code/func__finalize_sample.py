from __future__ import annotations
import heapq
import math
import random as rnd
from functools import partial
from itertools import islice
from dask.bag.core import Bag
def _finalize_sample(reduce_iter, k):
    sample = reduce_iter[0]
    if len(sample) < k:
        raise ValueError('Sample larger than population')
    return sample