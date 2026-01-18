from .functions import defun, defun_wrapped
def RJ_calc(ctx, x, y, z, p, r, integration):
    """
    With integration == 0, computes RJ only using Carlson's algorithm
    (may be wrong for some values).
    With integration == 1, uses an initial integration to make sure
    Carlson's algorithm is correct.
    With integration == 2, uses only integration.
    """
    if not (ctx.isnormal(x) and ctx.isnormal(y) and ctx.isnormal(z) and ctx.isnormal(p)):
        if ctx.isnan(x) or ctx.isnan(y) or ctx.isnan(z) or ctx.isnan(p):
            return x * y * z
        if ctx.isinf(x) or ctx.isinf(y) or ctx.isinf(z) or ctx.isinf(p):
            return ctx.zero
    if not p:
        return ctx.inf
    if (not x) + (not y) + (not z) > 1:
        return ctx.inf
    initial_integral = ctx.zero
    if integration >= 1:
        ok = x.real >= 0 and y.real >= 0 and (z.real >= 0) and (p.real > 0)
        if not ok:
            if x == p or y == p or z == p:
                ok = True
        if not ok:
            if p.imag != 0 or p.real >= 0:
                if x.imag == 0 and x.real >= 0 and (ctx.conj(y) == z):
                    ok = True
                if y.imag == 0 and y.real >= 0 and (ctx.conj(x) == z):
                    ok = True
                if z.imag == 0 and z.real >= 0 and (ctx.conj(x) == y):
                    ok = True
        if not ok or integration == 2:
            N = ctx.ceil(-min(x.real, y.real, z.real, p.real)) + 1
            if all((t.imag >= 0 or t.real > 0 for t in [x, y, z, p])):
                margin = ctx.j
            elif all((t.imag < 0 or t.real > 0 for t in [x, y, z, p])):
                margin = -ctx.j
            else:
                margin = 1
                for t in [x, y, z, p]:
                    if t.imag >= 0 or t.real > 0:
                        continue
                    margin = min(margin, abs(t.imag) * 0.5)
                margin *= ctx.j
            N += margin
            F = lambda t: 1 / (ctx.sqrt(t + x) * ctx.sqrt(t + y) * ctx.sqrt(t + z) * (t + p))
            if integration == 2:
                return 1.5 * ctx.quadsubdiv(F, [0, N, ctx.inf])
            initial_integral = 1.5 * ctx.quadsubdiv(F, [0, N])
            x += N
            y += N
            z += N
            p += N
    xm, ym, zm, pm = (x, y, z, p)
    A0 = Am = (x + y + z + 2 * p) / 5
    delta = (p - x) * (p - y) * (p - z)
    Q = ctx.root(0.25 * r, -6) * max(abs(A0 - x), abs(A0 - y), abs(A0 - z), abs(A0 - p))
    g = ctx.mpf(0.25)
    pow4 = ctx.one
    S = 0
    while 1:
        sx = ctx.sqrt(xm)
        sy = ctx.sqrt(ym)
        sz = ctx.sqrt(zm)
        sp = ctx.sqrt(pm)
        lm = sx * sy + sx * sz + sy * sz
        Am1 = (Am + lm) * g
        xm = (xm + lm) * g
        ym = (ym + lm) * g
        zm = (zm + lm) * g
        pm = (pm + lm) * g
        dm = (sp + sx) * (sp + sy) * (sp + sz)
        em = delta * pow4 ** 3 / dm ** 2
        if pow4 * Q < abs(Am):
            break
        T = RC_calc(ctx, ctx.one, ctx.one + em, r) * pow4 / dm
        S += T
        pow4 *= g
        Am = Am1
    t = pow4 / Am
    X = (A0 - x) * t
    Y = (A0 - y) * t
    Z = (A0 - z) * t
    P = (-X - Y - Z) / 2
    E2 = X * Y + X * Z + Y * Z - 3 * P ** 2
    E3 = X * Y * Z + 2 * E2 * P + 4 * P ** 3
    E4 = (2 * X * Y * Z + E2 * P + 3 * P ** 3) * P
    E5 = X * Y * Z * P ** 2
    P = 24024 - 5148 * E2 + 2457 * E2 ** 2 + 4004 * E3 - 4158 * E2 * E3 - 3276 * E4 + 2772 * E5
    Q = 24024
    v1 = pow4 * ctx.power(Am, -1.5) * P / Q
    v2 = 6 * S
    return initial_integral + v1 + v2