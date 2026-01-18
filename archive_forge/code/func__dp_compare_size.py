import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _dp_compare_size(cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs, i1_cut_i2_wo_output, memory_limit, cntrct1, cntrct2):
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on the size of the intermediate tensor created, rather than the number of
    operations, and so calculates that first.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = helpers.compute_size_by_dict(i, size_dict)
    cost = max(cost1, cost2, mem)
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (cntrct1, cntrct2))