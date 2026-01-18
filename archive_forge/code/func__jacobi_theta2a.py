from .functions import defun, defun_wrapped
@defun
def _jacobi_theta2a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for minimum (2*n+1)*log(q).real - 2* ctx._im(z)
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=n0, inf) +
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n, n0-1, -inf)
    """
    n = n0 = int(ctx._im(z) / ctx._re(ctx.log(q)) - 1 / 2)
    e2 = ctx.expj(2 * z)
    e = e0 = ctx.expj((2 * n + 1) * z)
    a = q ** (n * n + n)
    term = a * e
    s = term
    eps1 = ctx.eps * abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q ** (n * n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2 * z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q ** (n * n + n) * e
        if abs(term) < eps1:
            break
        s += term
    s = s * ctx.nthroot(q, 4)
    return s