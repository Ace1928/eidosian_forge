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
def _fix_ssa_vars(blocks, varname, defmap, cfg, df_plus, cache_list_vars):
    """Rewrite all uses to ``varname`` given the definition map
    """
    states = _make_states(blocks)
    states['varname'] = varname
    states['defmap'] = defmap
    states['phimap'] = phimap = defaultdict(list)
    states['cfg'] = cfg
    states['phi_locations'] = _compute_phi_locations(df_plus, defmap)
    newblocks = _run_block_rewrite(blocks, states, _FixSSAVars(cache_list_vars))
    for label, philist in phimap.items():
        curblk = newblocks[label]
        curblk.body = philist + curblk.body
    return newblocks