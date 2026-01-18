from __future__ import annotations
import heapq
import math
import random as rnd
from functools import partial
from itertools import islice
from dask.bag.core import Bag
def _weighted_sampling_without_replacement(population, weights, k):
    """
    Source:
        Weighted random sampling with a reservoir, Pavlos S. Efraimidis, Paul G. Spirakis
    """
    elt = [(math.log(rnd.random()) / weights[i], i) for i in range(len(weights))]
    return [population[x[1]] for x in heapq.nlargest(k, elt)]