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
def _run_block_rewrite(blocks, states, handler):
    newblocks = {}
    for label, blk in blocks.items():
        _logger.debug('==== SSA block rewrite pass on %s', label)
        newblk = ir.Block(scope=blk.scope, loc=blk.loc)
        newbody = []
        states['label'] = label
        states['block'] = blk
        for stmt in _run_ssa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks