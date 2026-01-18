from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class Ridder:
    """
    1d-solver generating pairs of approximative root and error.

    Ridders' method to find a root of f in [a, b].
    Is told to perform as well as Brent's method while being simpler.

    Pro:

    * very fast
    * simpler than Brent's method

    Contra:

    * two function evaluations per step
    * has problems with multiple roots
    * needs sign change

    http://en.wikipedia.org/wiki/Ridders'_method
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        self.f = f
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.x1 = x0[0]
        self.x2 = x0[1]
        self.verbose = kwargs['verbose']
        self.tol = kwargs['tol']

    def __iter__(self):
        ctx = self.ctx
        f = self.f
        x1 = self.x1
        fx1 = f(x1)
        x2 = self.x2
        fx2 = f(x2)
        while True:
            x3 = 0.5 * (x1 + x2)
            fx3 = f(x3)
            x4 = x3 + (x3 - x1) * ctx.sign(fx1 - fx2) * fx3 / ctx.sqrt(fx3 ** 2 - fx1 * fx2)
            fx4 = f(x4)
            if abs(fx4) < self.tol:
                if self.verbose:
                    print('canceled with f(x4) =', fx4)
                yield (x4, abs(x1 - x2))
                break
            if fx4 * fx2 < 0:
                x1 = x4
                fx1 = fx4
            else:
                x2 = x4
                fx2 = fx4
            error = abs(x1 - x2)
            yield ((x1 + x2) / 2, error)