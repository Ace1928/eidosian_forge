from .functions import defun, defun_wrapped
@defun
def _djacobi_theta3a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    djtheta3(z, q, nd) = (2*j)**nd *
      Sum(q**(n*n) * n**nd * exp(j*2*n*z), n, -inf, inf)
    max term for minimum n*abs(log(q).real) + ctx._im(z)
    """
    n = n0 = int(-ctx._im(z) / abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2 * z)
    e = e0 = ctx.expj(2 * n * z)
    a = q ** (n * n) * e
    s = term = n ** nd * a
    if n != 0:
        eps1 = ctx.eps * abs(term)
    else:
        eps1 = ctx.eps * abs(a)
    while 1:
        n += 1
        e = e * e2
        a = q ** (n * n) * e
        term = n ** nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2 * z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        a = q ** (n * n) * e
        term = n ** nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    return (2 * ctx.j) ** nd * s