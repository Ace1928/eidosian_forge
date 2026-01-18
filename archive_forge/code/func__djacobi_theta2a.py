from .functions import defun, defun_wrapped
@defun
def _djacobi_theta2a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    dtheta(2, z, q, nd) =
    j* q**1/4 * Sum(q**(n*n + n) * (2*n+1)*exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for (2*n0+1)*log(q).real - 2* ctx._im(z) ~= 0
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    """
    n = n0 = int(ctx._im(z) / ctx._re(ctx.log(q)) - 1 / 2)
    e2 = ctx.expj(2 * z)
    e = e0 = ctx.expj((2 * n + 1) * z)
    a = q ** (n * n + n)
    term = (2 * n + 1) ** nd * a * e
    s = term
    eps1 = ctx.eps * abs(term)
    while 1:
        n += 1
        e = e * e2
        term = (2 * n + 1) ** nd * q ** (n * n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2 * z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = (2 * n + 1) ** nd * q ** (n * n + n) * e
        if abs(term) < eps1:
            break
        s += term
    return ctx.j ** nd * s * ctx.nthroot(q, 4)