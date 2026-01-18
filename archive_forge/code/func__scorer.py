from .functions import defun, defun_wrapped
def _scorer(ctx, z, which, kwargs):
    z = ctx.convert(z)
    if ctx.isinf(z):
        if z == ctx.inf:
            if which == 0:
                return 1 / z
            if which == 1:
                return z
        if z == ctx.ninf:
            return 1 / z
        raise ValueError('essential singularity')
    if z:
        extraprec = max(0, int(1.5 * ctx.mag(z)))
    else:
        extraprec = 0
    if kwargs.get('derivative'):
        raise NotImplementedError
    try:
        if ctx.mag(z) > 3:
            if which == 0 and abs(ctx.arg(z)) < ctx.pi / 3 * 0.999:

                def h():
                    return (([ctx.pi, z], [-1, -1], [], [], [(1, 3), (2, 3), 1], [], 9 / z ** 3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
            if which == 1 and abs(ctx.arg(-z)) < 2 * ctx.pi / 3 * 0.999:

                def h():
                    return (([-ctx.pi, z], [-1, -1], [], [], [(1, 3), (2, 3), 1], [], 9 / z ** 3),)
                return ctx.hypercomb(h, [], maxterms=ctx.prec, force_series=True)
    except ctx.NoConvergence:
        pass

    def h():
        A = ctx.airybi(z, **kwargs) / 3
        B = -2 * ctx.pi
        if which == 1:
            A *= 2
            B *= -1
        ctx.prec += extraprec
        w = z ** 3 / 9
        ctx.prec -= extraprec
        T1 = ([A], [1], [], [], [], [], 0)
        T2 = ([B, z], [-1, 2], [], [], [1], [ctx.mpq_4_3, ctx.mpq_5_3], w)
        return (T1, T2)
    return ctx.hypercomb(h, [], **kwargs)