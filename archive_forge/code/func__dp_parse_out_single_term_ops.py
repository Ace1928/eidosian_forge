import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts):
    """Take ``inputs`` and parse for single term index operations, i.e. where
    an index appears on one tensor and nowhere else.

    If a term is completely reduced to a scalar in this way it can be removed
    to ``inputs_done``. If only some indices can be summed then add a 'single
    term contraction' that will perform this summation.
    """
    i_single = {i for i, c in enumerate(all_inds) if ind_counts[c] == 1}
    inputs_parsed, inputs_done, inputs_contractions = ([], [], [])
    for j, i in enumerate(inputs):
        i_reduced = i - i_single
        if not i_reduced:
            inputs_done.append((j,))
        else:
            inputs_parsed.append(i_reduced)
            inputs_contractions.append((j,) if i_reduced != i else j)
    return (inputs_parsed, inputs_done, inputs_contractions)