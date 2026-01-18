import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
def _process_basic_block(bb_lines):
    bb_lines = _move_and_group_decref_after_all_increfs(bb_lines)
    bb_lines = _prune_redundant_refct_ops(bb_lines)
    return bb_lines