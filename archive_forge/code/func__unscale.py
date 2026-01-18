import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
from collections import namedtuple
def _unscale(x, C, b_scale):
    """
    Converts solution to _autoscale problem -> solution to original problem.
    """
    try:
        n = len(C)
    except TypeError:
        n = len(x)
    return x[:n] * b_scale * C