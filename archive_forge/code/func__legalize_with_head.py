from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _legalize_with_head(blk):
    """Given *blk*, the head block of the with-context, check that it doesn't
    do anything else.
    """
    counters = defaultdict(int)
    for stmt in blk.body:
        counters[type(stmt)] += 1
    if counters.pop(ir.EnterWith) != 1:
        raise errors.CompilerError("with's head-block must have exactly 1 ENTER_WITH", loc=blk.loc)
    if counters.pop(ir.Jump, 0) != 1:
        raise errors.CompilerError("with's head-block must have exactly 1 JUMP", loc=blk.loc)
    counters.pop(ir.Del, None)
    if counters:
        raise errors.CompilerError("illegal statements in with's head-block", loc=blk.loc)