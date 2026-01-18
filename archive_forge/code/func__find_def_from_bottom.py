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
def _find_def_from_bottom(self, states, label, loc):
    """Find definition from within the block at ``label``.
        """
    _logger.debug('find_def_from_bottom label %r', label)
    defmap = states['defmap']
    defs = defmap[label]
    if defs:
        lastdef = defs[-1]
        return lastdef
    else:
        return self._find_def_from_top(states, label, loc=loc)