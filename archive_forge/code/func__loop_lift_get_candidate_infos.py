from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _loop_lift_get_candidate_infos(cfg, blocks, livemap):
    """
    Returns information on looplifting candidates.
    """
    loops = _extract_loop_lifting_candidates(cfg, blocks)
    loopinfos = []
    for loop in loops:
        [callfrom] = loop.entries
        an_exit = next(iter(loop.exits))
        if len(loop.exits) > 1:
            [(returnto, _)] = cfg.successors(an_exit)
        else:
            returnto = an_exit
        local_block_ids = set(loop.body) | set(loop.entries) | set(loop.exits)
        inputs, outputs = find_region_inout_vars(blocks=blocks, livemap=livemap, callfrom=callfrom, returnto=returnto, body_block_ids=local_block_ids)
        lli = _loop_lift_info(loop=loop, inputs=inputs, outputs=outputs, callfrom=callfrom, returnto=returnto)
        loopinfos.append(lli)
    return loopinfos