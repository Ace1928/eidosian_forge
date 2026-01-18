from .functions import defun, defun_wrapped
@defun_wrapped
def _coulomb_chi(ctx, l, eta, _cache={}):
    if (l, eta) in _cache and _cache[l, eta][0] >= ctx.prec:
        return _cache[l, eta][1]

    def terms():
        l2 = -l - 1
        jeta = ctx.j * eta
        return [ctx.loggamma(1 + l + jeta) * -0.5j, ctx.loggamma(1 + l - jeta) * 0.5j, ctx.loggamma(1 + l2 + jeta) * 0.5j, ctx.loggamma(1 + l2 - jeta) * -0.5j, -(l + 0.5) * ctx.pi]
    v = ctx.sum_accurately(terms, 1)
    _cache[l, eta] = (ctx.prec, v)
    return v