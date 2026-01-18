import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _get_class_vars_type(self):
    """Helper method to be able to pass needed vars to _compute_subset.

        Needs to be implemented by subclasses."""
    pass