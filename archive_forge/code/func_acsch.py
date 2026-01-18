from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def acsch(ctx, z):
    return ctx.asinh(ctx.one / z)