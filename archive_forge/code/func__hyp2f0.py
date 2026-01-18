from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp2f0(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    try:
        kwargsb = kwargs.copy()
        kwargsb['maxterms'] = kwargsb.get('maxterms', ctx.prec)
        return ctx.hypsum(2, 0, (atype, btype), [a, b], z, **kwargsb)
    except ctx.NoConvergence:
        if kwargs.get('force_series'):
            raise
        pass

    def h(a, b):
        w = ctx.sinpi(b)
        rz = -1 / z
        T1 = ([ctx.pi, w, rz], [1, -1, a], [], [a - b + 1, b], [a], [b], rz)
        T2 = ([-ctx.pi, w, rz], [1, -1, 1 + a - b], [], [a, 2 - b], [a - b + 1], [2 - b], rz)
        return (T1, T2)
    return ctx.hypercomb(h, [a, 1 + a - b], **kwargs)