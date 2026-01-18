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
def _run_ssa(blocks):
    """Run SSA reconstruction on IR blocks of a function.
    """
    if not blocks:
        return {}
    cfg = compute_cfg_from_blocks(blocks)
    df_plus = _iterated_domfronts(cfg)
    violators = _find_defs_violators(blocks, cfg)
    cache_list_vars = _CacheListVars()
    for varname in violators:
        _logger.debug('Fix SSA violator on var %s', varname)
        blocks, defmap = _fresh_vars(blocks, varname)
        _logger.debug('Replaced assignments: %s', pformat(defmap))
        blocks = _fix_ssa_vars(blocks, varname, defmap, cfg, df_plus, cache_list_vars)
    cfg_post = compute_cfg_from_blocks(blocks)
    if cfg_post != cfg:
        raise errors.CompilerError('CFG mutated in SSA pass')
    return blocks