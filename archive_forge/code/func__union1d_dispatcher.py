import functools
import numpy as np
from numpy.core import overrides
def _union1d_dispatcher(ar1, ar2):
    return (ar1, ar2)