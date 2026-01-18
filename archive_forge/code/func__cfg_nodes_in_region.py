from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _cfg_nodes_in_region(cfg, region_begin, region_end):
    """Find the set of CFG nodes that are in the given region
    """
    region_nodes = set()
    stack = [region_begin]
    while stack:
        tos = stack.pop()
        succlist = list(cfg.successors(tos))
        if succlist:
            succs, _ = zip(*succlist)
            nodes = set([node for node in succs if node not in region_nodes and node != region_end])
            stack.extend(nodes)
            region_nodes |= nodes
    return region_nodes