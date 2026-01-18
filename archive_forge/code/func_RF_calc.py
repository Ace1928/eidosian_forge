from .functions import defun, defun_wrapped
def RF_calc(ctx, x, y, z, r):
    if y == z:
        return RC_calc(ctx, x, y, r)
    if x == z:
        return RC_calc(ctx, y, x, r)
    if x == y:
        return RC_calc(ctx, z, x, r)
    if not (ctx.isnormal(x) and ctx.isnormal(y) and ctx.isnormal(z)):
        if ctx.isnan(x) or ctx.isnan(y) or ctx.isnan(z):
            return x * y * z
        if ctx.isinf(x) or ctx.isinf(y) or ctx.isinf(z):
            return ctx.zero
    xm, ym, zm = (x, y, z)
    A0 = Am = (x + y + z) / 3
    Q = ctx.root(3 * r, -6) * max(abs(A0 - x), abs(A0 - y), abs(A0 - z))
    g = ctx.mpf(0.25)
    pow4 = ctx.one
    while 1:
        xs = ctx.sqrt(xm)
        ys = ctx.sqrt(ym)
        zs = ctx.sqrt(zm)
        lm = xs * ys + xs * zs + ys * zs
        Am1 = (Am + lm) * g
        xm, ym, zm = ((xm + lm) * g, (ym + lm) * g, (zm + lm) * g)
        if pow4 * Q < abs(Am):
            break
        Am = Am1
        pow4 *= g
    t = pow4 / Am
    X = (A0 - x) * t
    Y = (A0 - y) * t
    Z = -X - Y
    E2 = X * Y - Z ** 2
    E3 = X * Y * Z
    return ctx.power(Am, -0.5) * (9240 - 924 * E2 + 385 * E2 ** 2 + 660 * E3 - 630 * E2 * E3) / 9240