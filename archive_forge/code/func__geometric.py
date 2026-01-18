from __future__ import annotations
import heapq
import math
import random as rnd
from functools import partial
from itertools import islice
from dask.bag.core import Bag
def _geometric(p):
    return int(math.log(rnd.uniform(0, 1)) / math.log(1 - p)) + 1