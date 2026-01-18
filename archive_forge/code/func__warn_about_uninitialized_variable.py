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
def _warn_about_uninitialized_variable(varname, loc):
    if config.ALWAYS_WARN_UNINIT_VAR:
        warnings.warn(errors.NumbaWarning(f'Detected uninitialized variable {varname}', loc=loc))