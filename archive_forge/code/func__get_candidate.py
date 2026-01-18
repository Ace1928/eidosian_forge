import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = either & output | two & dim_ref_counts[3] | one & dim_ref_counts[2]
    cost = cost_fn(helpers.compute_size_by_dict(k12, sizes), footprints[k1], footprints[k2], k12, k1, k2)
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = (k2, id2, k1, id1)
    cost = (cost, id2, id1)
    return (cost, k1, k2, k12)