import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _dp_compare_flops(cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs, i1_cut_i2_wo_output, memory_limit, cntrct1, cntrct2):
    """Performs the inner comparison of whether the two subgraphs (the bitmaps
    ``s1`` and ``s2``) should be merged and added to the dynamic programming
    search. Will skip for a number of reasons:

    1. If the number of operations to form ``s = s1 | s2`` including previous
       contractions is above the cost-cap.
    2. If we've already found a better way of making ``s``.
    3. If the intermediate tensor corresponding to ``s`` is going to break the
       memory limit.
    """
    cost = cost1 + cost2 + helpers.compute_size_by_dict(i1_union_i2, size_dict)
    if cost <= cost_cap:
        s = s1 | s2
        if s not in xn or cost < xn[s][1]:
            i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
            mem = helpers.compute_size_by_dict(i, size_dict)
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (cntrct1, cntrct2))