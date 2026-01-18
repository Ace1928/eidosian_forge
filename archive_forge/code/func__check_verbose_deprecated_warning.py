import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def _check_verbose_deprecated_warning(verbose):
    """Raises a warning when verbose is not the default value."""
    if verbose != 'deprecated':
        warnings.warn('The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.', UserWarning)
        return verbose
    return False