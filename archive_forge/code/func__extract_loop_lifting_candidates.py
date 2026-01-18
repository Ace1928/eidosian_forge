from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _extract_loop_lifting_candidates(cfg, blocks):
    """
    Returns a list of loops that are candidate for loop lifting
    """

    def same_exit_point(loop):
        """all exits must point to the same location"""
        outedges = set()
        for k in loop.exits:
            succs = set((x for x, _ in cfg.successors(k)))
            if not succs:
                _logger.debug('return-statement in loop.')
                return False
            outedges |= succs
        ok = len(outedges) == 1
        _logger.debug('same_exit_point=%s (%s)', ok, outedges)
        return ok

    def one_entry(loop):
        """there is one entry"""
        ok = len(loop.entries) == 1
        _logger.debug('one_entry=%s', ok)
        return ok

    def cannot_yield(loop):
        """cannot have yield inside the loop"""
        insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
        for blk in map(blocks.__getitem__, insiders):
            for inst in blk.body:
                if isinstance(inst, ir.Assign):
                    if isinstance(inst.value, ir.Yield):
                        _logger.debug('has yield')
                        return False
        _logger.debug('no yield')
        return True
    _logger.info('finding looplift candidates')
    candidates = []
    for loop in find_top_level_loops(cfg):
        _logger.debug('top-level loop: %s', loop)
        if same_exit_point(loop) and one_entry(loop) and cannot_yield(loop) and (cfg.entry_point() not in loop.entries):
            candidates.append(loop)
            _logger.debug('add candidate: %s', loop)
    return candidates