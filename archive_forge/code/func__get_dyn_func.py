import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
def _get_dyn_func(**jit_args):
    code = '\n        def dyn_func(x):\n            res = 0\n            for i in range(x):\n                res += x\n            return res\n        '
    ns = {}
    exec(code.strip(), ns)
    return jit(**jit_args)(ns['dyn_func'])