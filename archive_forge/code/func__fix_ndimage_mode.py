import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _fix_ndimage_mode(mode):
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    return grid_modes.get(mode, mode)