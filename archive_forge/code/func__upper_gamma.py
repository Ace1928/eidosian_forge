from .functions import defun, defun_wrapped
@defun
def _upper_gamma(ctx, z, a, regularized=False):
    if ctx.isint(z):
        try:
            if regularized:
                if ctx.isnpint(z):
                    return type(z)(ctx.zero)
                orig = ctx.prec
                try:
                    ctx.prec += 10
                    return ctx._gamma_upper_int(z, a) / ctx.gamma(z)
                finally:
                    ctx.prec = orig
            else:
                return ctx._gamma_upper_int(z, a)
        except NotImplementedError:
            pass
    if z == 2 and a == -1:
        return (z + a) * 0
    if z == 3 and (a == -1 - 1j or a == -1 + 1j):
        return (z + a) * 0
    nega = ctx.fneg(a, exact=True)
    G = [z] * regularized
    try:

        def h(z):
            r = z - 1
            return [([ctx.exp(nega), a], [1, r], [], G, [1, -r], [], 1 / nega)]
        return ctx.hypercomb(h, [z], force_series=True)
    except ctx.NoConvergence:

        def h(z):
            T1 = ([], [1, z - 1], [z], G, [], [], 0)
            T2 = ([-ctx.exp(nega), a, z], [1, z, -1], [], G, [1], [1 + z], a)
            return (T1, T2)
        return ctx.hypercomb(h, [z])