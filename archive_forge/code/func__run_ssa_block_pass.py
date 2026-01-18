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
def _run_ssa_block_pass(states, blk, handler):
    _logger.debug('Running %s', handler)
    for stmt in blk.body:
        _logger.debug('on stmt: %s', stmt)
        if isinstance(stmt, ir.Assign):
            ret = handler.on_assign(states, stmt)
        else:
            ret = handler.on_other(states, stmt)
        if ret is not stmt and ret is not None:
            _logger.debug('replaced with: %s', ret)
        yield ret