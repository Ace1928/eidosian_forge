from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def _find_interval(self, v):
    m = len(self.intervals)
    i = 0
    while i < m - 1 and self.intervals[i] <= v:
        i += 1
    return i