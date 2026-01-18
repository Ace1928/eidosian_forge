import builtins
from warnings import catch_warnings, simplefilter
import numpy as np
from operator import index
from collections import namedtuple
def _bincount(x, weights):
    if np.iscomplexobj(weights):
        a = np.bincount(x, np.real(weights))
        b = np.bincount(x, np.imag(weights))
        z = a + b * 1j
    else:
        z = np.bincount(x, weights)
    return z