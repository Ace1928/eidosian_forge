from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun_wrapped
def _hyp1f0(ctx, a, z):
    return (1 - z) ** (-a)