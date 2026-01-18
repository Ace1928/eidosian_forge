import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
def _trial_greedy_ssa_path_and_cost(r, inputs, output, size_dict, choose_fn, cost_fn):
    """A single, repeatable, greedy trial run. Returns ``ssa_path`` and cost.
    """
    if r == 0:
        choose_fn = None
    random_seed(r)
    ssa_path = paths.ssa_greedy_optimize(inputs, output, size_dict, choose_fn, cost_fn)
    cost, size = ssa_path_compute_cost(ssa_path, inputs, output, size_dict)
    return (ssa_path, cost, size)