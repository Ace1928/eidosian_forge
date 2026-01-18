from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _loop_lift_prepare_loop_func(loopinfo, blocks):
    """
    Inplace transform loop blocks for use as lifted loop.
    """
    entry_block = blocks[loopinfo.callfrom]
    scope = entry_block.scope
    loc = entry_block.loc
    firstblk = min(blocks) - 1
    blocks[firstblk] = ir_utils.fill_callee_prologue(block=ir.Block(scope=scope, loc=loc), inputs=loopinfo.inputs, label_next=loopinfo.callfrom)
    blocks[loopinfo.returnto] = ir_utils.fill_callee_epilogue(block=ir.Block(scope=scope, loc=loc), outputs=loopinfo.outputs)