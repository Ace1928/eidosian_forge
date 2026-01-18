from ..libmp.backend import xrange
import math
import cmath
def _polyexp(ctx, n, x, extra=False):

    def _terms():
        if extra:
            yield ctx.sincpi(n)
        t = x
        k = 1
        while 1:
            yield (k ** n * t)
            k += 1
            t = t * x / k
    return ctx.sum_accurately(_terms, check_step=4)