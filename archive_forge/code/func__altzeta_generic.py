from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def _altzeta_generic(ctx, s):
    if s == 1:
        return ctx.ln2 + 0 * s
    return -ctx.powm1(2, 1 - s) * ctx.zeta(s)