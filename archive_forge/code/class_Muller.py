from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class Muller:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0, x1 and x2 close to the root.
    x1 defaults to x0 + 0.25; x2 to x1 + 0.25.
    Uses Muller's method that converges towards complex roots.

    Pro:

    * converges fast (somewhat faster than secant)
    * can find complex roots

    Contra:

    * converges slowly for multiple roots
    * may have complex values for real starting points and real roots

    http://en.wikipedia.org/wiki/Muller's_method
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
            self.x1 = self.x0 + 0.25
            self.x2 = self.x1 + 0.25
        elif len(x0) == 2:
            self.x0 = x0[0]
            self.x1 = x0[1]
            self.x2 = self.x1 + 0.25
        elif len(x0) == 3:
            self.x0 = x0[0]
            self.x1 = x0[1]
            self.x2 = x0[2]
        else:
            raise ValueError('expected 1, 2 or 3 starting points, got %i' % len(x0))
        self.f = f
        self.verbose = kwargs['verbose']

    def __iter__(self):
        f = self.f
        x0 = self.x0
        x1 = self.x1
        x2 = self.x2
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
        while True:
            fx2x1 = (fx1 - fx2) / (x1 - x2)
            fx2x0 = (fx0 - fx2) / (x0 - x2)
            fx1x0 = (fx0 - fx1) / (x0 - x1)
            w = fx2x1 + fx2x0 - fx1x0
            fx2x1x0 = (fx1x0 - fx2x1) / (x0 - x2)
            if w == 0 and fx2x1x0 == 0:
                if self.verbose:
                    print('canceled with')
                    print('x0 =', x0, ', x1 =', x1, 'and x2 =', x2)
                break
            x0 = x1
            fx0 = fx1
            x1 = x2
            fx1 = fx2
            r = self.ctx.sqrt(w ** 2 - 4 * fx2 * fx2x1x0)
            if abs(w - r) > abs(w + r):
                r = -r
            x2 -= 2 * fx2 / (w + r)
            fx2 = f(x2)
            error = abs(x2 - x1)
            yield (x2, error)