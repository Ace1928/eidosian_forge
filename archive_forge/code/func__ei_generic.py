from .functions import defun, defun_wrapped
@defun_wrapped
def _ei_generic(ctx, z):
    if z == ctx.inf:
        return z
    if z == ctx.ninf:
        return ctx.zero
    if ctx.mag(z) > 1:
        try:
            r = ctx.one / z
            v = ctx.exp(z) * ctx.hyper([1, 1], [], r, maxterms=ctx.prec, force_series=True) / z
            im = ctx._im(z)
            if im > 0:
                v += ctx.pi * ctx.j
            if im < 0:
                v -= ctx.pi * ctx.j
            return v
        except ctx.NoConvergence:
            pass
    v = z * ctx.hyp2f2(1, 1, 2, 2, z) + ctx.euler
    if ctx._im(z):
        v += 0.5 * (ctx.log(z) - ctx.log(ctx.one / z))
    else:
        v += ctx.log(abs(z))
    return v