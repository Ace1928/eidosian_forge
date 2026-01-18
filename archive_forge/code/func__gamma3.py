from .functions import defun, defun_wrapped
@defun
def _gamma3(ctx, z, a, b, regularized=False):
    pole = ctx.isnpint(z)
    if regularized and pole:
        return ctx.zero
    try:
        ctx.prec += 15
        T1 = ctx.gammainc(z, a, regularized=regularized)
        T2 = ctx.gammainc(z, b, regularized=regularized)
        R = T1 - T2
        if ctx.mag(R) - max(ctx.mag(T1), ctx.mag(T2)) > -10:
            return R
        if not pole:
            T1 = ctx.gammainc(z, 0, b, regularized=regularized)
            T2 = ctx.gammainc(z, 0, a, regularized=regularized)
            R = T1 - T2
            if 1:
                return R
    finally:
        ctx.prec -= 15
    raise NotImplementedError