import functools
import numpy as np
from numpy.core import overrides
def _ediff1d_dispatcher(ary, to_end=None, to_begin=None):
    return (ary, to_end, to_begin)