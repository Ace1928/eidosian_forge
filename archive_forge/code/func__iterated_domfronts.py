import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
def _iterated_domfronts(cfg):
    """Compute the iterated dominance frontiers (DF+ in literatures).

    Returns a dictionary which maps block label to the set of labels of its
    iterated dominance frontiers.
    """
    domfronts = {k: set(vs) for k, vs in cfg.dominance_frontier().items()}
    keep_going = True
    while keep_going:
        keep_going = False
        for k, vs in domfronts.items():
            inner = reduce(operator.or_, [domfronts[v] for v in vs], set())
            if inner.difference(vs):
                vs |= inner
                keep_going = True
    return domfronts