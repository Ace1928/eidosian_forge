from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
def _hyp2f1_gosper(ctx, a, b, c, z, **kwargs):
    _a, _b, _c, _z = (a, b, c, z)
    orig = ctx.prec
    maxprec = kwargs.get('maxprec', 100 * orig)
    extra = 10
    while 1:
        ctx.prec = orig + extra
        z = ctx.convert(_z)
        d = ctx.mpf(0)
        e = ctx.mpf(1)
        f = ctx.mpf(0)
        k = 0
        abz = a * b * z
        ch = c * ctx.mpq_1_2
        c1h = (c + 1) * ctx.mpq_1_2
        nz = 1 - z
        g = z / nz
        abg = a * b * g
        cba = c - b - a
        z2 = z - 2
        tol = -ctx.prec - 10
        nstr = ctx.nstr
        nprint = ctx.nprint
        mag = ctx.mag
        maxmag = ctx.ninf
        while 1:
            kch = k + ch
            kakbz = (k + a) * (k + b) * z / (4 * (k + 1) * kch * (k + c1h))
            d1 = kakbz * (e - (k + cba) * d * g)
            e1 = kakbz * (d * abg + (k + c) * e)
            ft = d * (k * (cba * z + k * z2 - c) - abz) / (2 * kch * nz)
            f1 = f + e - ft
            maxmag = max(maxmag, mag(f1))
            if mag(f1 - f) < tol:
                break
            d, e, f = (d1, e1, f1)
            k += 1
        cancellation = maxmag - mag(f1)
        if cancellation < extra:
            break
        else:
            extra += cancellation
            if extra > maxprec:
                raise ctx.NoConvergence
    return f1