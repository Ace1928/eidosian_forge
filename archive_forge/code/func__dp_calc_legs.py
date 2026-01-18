import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2):
    """Calculates the effective outer indices of the intermediate tensor
    corresponding to the subgraph ``s``.
    """
    r = g & (all_tensors ^ s)
    if r:
        i_r = set.union(*_bitmap_select(r, inputs))
    else:
        i_r = set()
    i_contract = i1_cut_i2_wo_output - i_r
    return i1_union_i2 - i_contract