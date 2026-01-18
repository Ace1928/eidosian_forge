from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp2f3(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type), (b3, b3type) = b_s
    absz = abs(z)
    magz = ctx.mag(z)
    asymp_extraprec = z and magz // 2
    orig = ctx.prec
    can_use_asymptotic = not kwargs.get('force_series') and ctx.mag(absz) > 19 and (ctx.sqrt(absz) > 1.5 * orig)
    if can_use_asymptotic:
        try:
            try:
                ctx.prec += asymp_extraprec

                def h(a1, a2, b1, b2, b3):
                    X = ctx.mpq_1_2 * (a1 + a2 - b1 - b2 - b3 + ctx.mpq_1_2)
                    A2 = a1 + a2
                    B3 = b1 + b2 + b3
                    A = a1 * a2
                    B = b1 * b2 + b3 * b2 + b1 * b3
                    R = b1 * b2 * b3
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2 * (B - A + ctx.mpq_1_4 * (3 * A2 + B3 - 2) * (A2 - B3) - ctx.mpq_3_16)
                    c[2] = ctx.mpq_1_2 * c[1] ** 2 + ctx.mpq_1_16 * (-16 * (2 * A2 - 3) * (B - A) + 32 * R + 4 * (-8 * A2 ** 2 + 11 * A2 + 8 * A + B3 - 2) * (A2 - B3) - 3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (k - 2 * X - 3) * (k - 2 * X - 2 * b1 - 1) * (k - 2 * X - 2 * b2 - 1) * (k - 2 * X - 2 * b3 - 1)
                            uu2 = 4 * (k - 1) ** 3 - 6 * (4 * X + B3) * (k - 1) ** 2 + 2 * (24 * X ** 2 + 12 * B3 * X + 4 * B + B3 - 1) * (k - 1) - 32 * X ** 3 - 24 * B3 * X ** 2 - 4 * B - 8 * R - 4 * (4 * B + B3 - 1) * X + 2 * B3 - 1
                            uu3 = 5 * (k - 1) ** 2 + 2 * (-10 * X + A2 - 3 * B3 + 3) * (k - 1) + 2 * c[1]
                            c[k] = ctx.one / (2 * k) * (uu1 * c[k - 3] - uu2 * c[k - 2] + uu3 * c[k - 1])
                        w = c[k] * ctx.power(-z, -0.5 * k)
                        t1 = (-ctx.j) ** k * ctx.mpf(2) ** (-k) * w
                        t2 = ctx.j ** k * ctx.mpf(2) ** (-k) * w
                        if abs(t1) < 0.1 * ctx.eps:
                            break
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi * X + 2 * ctx.sqrt(-z)) * s1 + ctx.expj(-(ctx.pi * X + 2 * ctx.sqrt(-z))) * s2
                    T1 = ([0.5 * S, ctx.pi, -z], [1, -0.5, X], [b1, b2, b3], [a1, a2], [], [], 0)
                    T2 = ([-z], [-a1], [b1, b2, b3, a2 - a1], [a2, b1 - a1, b2 - a1, b3 - a1], [a1, a1 - b1 + 1, a1 - b2 + 1, a1 - b3 + 1], [a1 - a2 + 1], 1 / z)
                    T3 = ([-z], [-a2], [b1, b2, b3, a1 - a2], [a1, b1 - a2, b2 - a2, b3 - a2], [a2, a2 - b1 + 1, a2 - b2 + 1, a2 - b3 + 1], [-a1 + a2 + 1], 1 / z)
                    return (T1, T2, T3)
                v = ctx.hypercomb(h, [a1, a2, b1, b2, b3], force_series=True, maxterms=4 * ctx.prec)
                if sum((ctx._is_real_type(u) for u in [a1, a2, b1, b2, b3, z])) == 6:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig
    return ctx.hypsum(2, 3, (a1type, a2type, b1type, b2type, b3type), [a1, a2, b1, b2, b3], z, **kwargs)