from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp2f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    (c, ctype), = b_s
    if z == 1:
        convergent = ctx.re(c - a - b) > 0
        finite = ctx.isint(a) and a <= 0 or (ctx.isint(b) and b <= 0)
        zerodiv = ctx.isint(c) and c <= 0 and (not (ctx.isint(a) and c <= a <= 0 or (ctx.isint(b) and c <= b <= 0)))
        if (convergent or finite) and (not zerodiv):
            return ctx.gammaprod([c, c - a - b], [c - a, c - b], _infsign=True)
        return ctx.hyp2f1(a, b, c, 1 - ctx.eps * 2) * ctx.inf
    if not z:
        if c or a == 0 or b == 0:
            return 1 + z
        return ctx.nan
    if ctx.isint(c) and c <= 0:
        if ctx.isint(a) and c <= a <= 0 or (ctx.isint(b) and c <= b <= 0):
            pass
        else:
            return ctx.inf
    absz = abs(z)
    if absz <= 0.8 or (ctx.isint(a) and a <= 0 and (a >= -1000)) or (ctx.isint(b) and b <= 0 and (b >= -1000)):
        return ctx.hypsum(2, 1, (atype, btype, ctype), [a, b, c], z, **kwargs)
    orig = ctx.prec
    try:
        ctx.prec += 10
        if absz >= 1.3:

            def h(a, b):
                t = ctx.mpq_1 - c
                ab = a - b
                rz = 1 / z
                T1 = ([-z], [-a], [c, -ab], [b, c - a], [a, t + a], [ctx.mpq_1 + ab], rz)
                T2 = ([-z], [-b], [c, ab], [a, c - b], [b, t + b], [ctx.mpq_1 - ab], rz)
                return (T1, T2)
            v = ctx.hypercomb(h, [a, b], **kwargs)
        elif abs(1 - z) <= 0.75:

            def h(a, b):
                t = c - a - b
                ca = c - a
                cb = c - b
                rz = 1 - z
                T1 = ([], [], [c, t], [ca, cb], [a, b], [1 - t], rz)
                T2 = ([rz], [t], [c, a + b - c], [a, b], [ca, cb], [1 + t], rz)
                return (T1, T2)
            v = ctx.hypercomb(h, [a, b], **kwargs)
        elif abs(z / (z - 1)) <= 0.75:
            v = ctx.hyp2f1(a, c - b, c, z / (z - 1)) / (1 - z) ** a
        else:
            v = _hyp2f1_gosper(ctx, a, b, c, z, **kwargs)
    finally:
        ctx.prec = orig
    return +v