import functools
import numpy as np
from numpy.core import overrides
def _in1d_dispatcher(ar1, ar2, assume_unique=None, invert=None, *, kind=None):
    return (ar1, ar2)