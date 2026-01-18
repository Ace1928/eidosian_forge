import functools
import numpy as np
from numpy.core import overrides
def _unique_dispatcher(ar, return_index=None, return_inverse=None, return_counts=None, axis=None, *, equal_nan=None):
    return (ar,)