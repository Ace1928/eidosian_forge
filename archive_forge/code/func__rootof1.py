from ..libmp.backend import xrange
import math
import cmath
@defun
def _rootof1(ctx, k, n):
    k = int(k)
    n = int(n)
    k %= n
    if not k:
        return ctx.one
    elif 2 * k == n:
        return -ctx.one
    elif 4 * k == n:
        return ctx.j
    elif 4 * k == 3 * n:
        return -ctx.j
    return ctx.expjpi(2 * ctx.mpf(k) / n)