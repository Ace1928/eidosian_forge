from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
class ColorGradient:
    colors = ([0.4, 0.4, 0.4], [0.9, 0.9, 0.9])
    intervals = (0.0, 1.0)

    def __init__(self, *args):
        if len(args) == 2:
            self.colors = list(args)
            self.intervals = [0.0, 1.0]
        elif len(args) > 0:
            if len(args) % 2 != 0:
                raise ValueError('len(args) should be even')
            self.colors = [args[i] for i in range(1, len(args), 2)]
            self.intervals = [args[i] for i in range(0, len(args), 2)]
        assert len(self.colors) == len(self.intervals)

    def copy(self):
        c = ColorGradient()
        c.colors = [e[:] for e in self.colors]
        c.intervals = self.intervals[:]
        return c

    def _find_interval(self, v):
        m = len(self.intervals)
        i = 0
        while i < m - 1 and self.intervals[i] <= v:
            i += 1
        return i

    def _interpolate_axis(self, axis, v):
        i = self._find_interval(v)
        v = rinterpolate(self.intervals[i - 1], self.intervals[i], v)
        return interpolate(self.colors[i - 1][axis], self.colors[i][axis], v)

    def __call__(self, r, g, b):
        c = self._interpolate_axis
        return (c(0, r), c(1, g), c(2, b))