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
def _find_def_from_top(self, states, label, loc):
    """Find definition reaching block of ``label``.

        This method would look at all dominance frontiers.
        Insert phi node if necessary.
        """
    _logger.debug('find_def_from_top label %r', label)
    cfg = states['cfg']
    defmap = states['defmap']
    phimap = states['phimap']
    phi_locations = states['phi_locations']
    if label in phi_locations:
        scope = states['scope']
        loc = states['block'].loc
        freshvar = scope.redefine(states['varname'], loc=loc)
        phinode = ir.Assign(target=freshvar, value=ir.Expr.phi(loc=loc), loc=loc)
        _logger.debug('insert phi node %s at %s', phinode, label)
        defmap[label].insert(0, phinode)
        phimap[label].append(phinode)
        for pred, _ in cfg.predecessors(label):
            incoming_def = self._find_def_from_bottom(states, pred, loc=loc)
            _logger.debug('incoming_def %s', incoming_def)
            phinode.value.incoming_values.append(incoming_def.target)
            phinode.value.incoming_blocks.append(pred)
        return phinode
    else:
        idom = cfg.immediate_dominators()[label]
        if idom == label:
            _warn_about_uninitialized_variable(states['varname'], loc)
            return UndefinedVariable
        _logger.debug('idom %s from label %s', idom, label)
        return self._find_def_from_bottom(states, idom, loc=loc)