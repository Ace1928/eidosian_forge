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
def _compute_phi_locations(iterated_df, defmap):
    phi_locations = set()
    for deflabel, defstmts in defmap.items():
        if defstmts:
            phi_locations |= iterated_df[deflabel]
    return phi_locations