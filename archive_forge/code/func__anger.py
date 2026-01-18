from .functions import defun, defun_wrapped
def _anger(ctx, which, v, z, **kwargs):
    v = ctx._convert_param(v)[0]
    z = ctx.convert(z)

    def h(v):
        b = ctx.mpq_1_2
        u = v * b
        m = b * 3
        a1, a2, b1, b2 = (m - u, m + u, 1 - u, 1 + u)
        c, s = ctx.cospi_sinpi(u)
        if which == 0:
            A, B = ([b * z, s], [c])
        if which == 1:
            A, B = ([b * z, -c], [s])
        w = ctx.square_exp_arg(z, mult=-0.25)
        T1 = (A, [1, 1], [], [a1, a2], [1], [a1, a2], w)
        T2 = (B, [1], [], [b1, b2], [1], [b1, b2], w)
        return (T1, T2)
    return ctx.hypercomb(h, [v], **kwargs)