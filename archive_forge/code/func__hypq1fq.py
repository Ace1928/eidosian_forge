from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hypq1fq(ctx, p, q, a_s, b_s, z, **kwargs):
    """
    Evaluates 3F2, 4F3, 5F4, ...
    """
    a_s, a_types = zip(*a_s)
    b_s, b_types = zip(*b_s)
    a_s = list(a_s)
    b_s = list(b_s)
    absz = abs(z)
    ispoly = False
    for a in a_s:
        if ctx.isint(a) and a <= 0:
            ispoly = True
            break
    if absz < 1 or ispoly:
        try:
            return ctx.hypsum(p, q, a_types + b_types, a_s + b_s, z, **kwargs)
        except ctx.NoConvergence:
            if absz > 1.1 or ispoly:
                raise
    if z == 1:
        S = ctx.re(sum(b_s) - sum(a_s))
        if S <= 0:
            return ctx.hyper(a_s, b_s, 0.9, **kwargs) * ctx.inf
    if (p, q) == (3, 2) and abs(z - 1) < 0.05:
        a1, a2, a3 = a_s
        b1, b2 = b_s
        u = b1 + b2 - a3
        initial = ctx.gammaprod([b2 - a3, b1 - a3, a1, a2], [b2 - a3, b1 - a3, 1, u])

        def term(k, _cache={0: initial}):
            u = b1 + b2 - a3 + k
            if k in _cache:
                t = _cache[k]
            else:
                t = _cache[k - 1]
                t *= (b1 + k - a3 - 1) * (b2 + k - a3 - 1)
                t /= k * (u - 1)
                _cache[k] = t
            return t * ctx.hyp2f1(a1, a2, u, z)
        try:
            S = ctx.nsum(term, [0, ctx.inf], verbose=kwargs.get('verbose'), strict=kwargs.get('strict', True))
            return S * ctx.gammaprod([b1, b2], [a1, a2, a3])
        except ctx.NoConvergence:
            pass
    if absz < 1.1 and ctx._re(z) <= 1:

        def term(kk, _cache={0: ctx.one}):
            k = int(kk)
            if k != kk:
                t = z ** ctx.mpf(kk) / ctx.fac(kk)
                for a in a_s:
                    t *= ctx.rf(a, kk)
                for b in b_s:
                    t /= ctx.rf(b, kk)
                return t
            if k in _cache:
                return _cache[k]
            t = term(k - 1)
            m = k - 1
            for j in xrange(p):
                t *= a_s[j] + m
            for j in xrange(q):
                t /= b_s[j] + m
            t *= z
            t /= k
            _cache[k] = t
            return t
        sum_method = kwargs.get('sum_method', 'r+s+e')
        try:
            return ctx.nsum(term, [0, ctx.inf], verbose=kwargs.get('verbose'), strict=kwargs.get('strict', True), method=sum_method.replace('e', ''))
        except ctx.NoConvergence:
            if 'e' not in sum_method:
                raise
            pass
        if kwargs.get('verbose'):
            print('Attempting Euler-Maclaurin summation')
        '\n        Somewhat slower version (one diffs_exp for each factor).\n        However, this would be faster with fast direct derivatives\n        of the gamma function.\n\n        def power_diffs(k0):\n            r = 0\n            l = ctx.log(z)\n            while 1:\n                yield z**ctx.mpf(k0) * l**r\n                r += 1\n\n        def loggamma_diffs(x, reciprocal=False):\n            sign = (-1) ** reciprocal\n            yield sign * ctx.loggamma(x)\n            i = 0\n            while 1:\n                yield sign * ctx.psi(i,x)\n                i += 1\n\n        def hyper_diffs(k0):\n            b2 = b_s + [1]\n            A = [ctx.diffs_exp(loggamma_diffs(a+k0)) for a in a_s]\n            B = [ctx.diffs_exp(loggamma_diffs(b+k0,True)) for b in b2]\n            Z = [power_diffs(k0)]\n            C = ctx.gammaprod([b for b in b2], [a for a in a_s])\n            for d in ctx.diffs_prod(A + B + Z):\n                v = C * d\n                yield v\n        '

        def log_diffs(k0):
            b2 = b_s + [1]
            yield (sum((ctx.loggamma(a + k0) for a in a_s)) - sum((ctx.loggamma(b + k0) for b in b2)) + k0 * ctx.log(z))
            i = 0
            while 1:
                v = sum((ctx.psi(i, a + k0) for a in a_s)) - sum((ctx.psi(i, b + k0) for b in b2))
                if i == 0:
                    v += ctx.log(z)
                yield v
                i += 1

        def hyper_diffs(k0):
            C = ctx.gammaprod([b for b in b_s], [a for a in a_s])
            for d in ctx.diffs_exp(log_diffs(k0)):
                v = C * d
                yield v
        tol = ctx.eps / 1024
        prec = ctx.prec
        try:
            trunc = 50 * ctx.dps
            ctx.prec += 20
            for i in xrange(5):
                head = ctx.fsum((term(k) for k in xrange(trunc)))
                tail, err = ctx.sumem(term, [trunc, ctx.inf], tol=tol, adiffs=hyper_diffs(trunc), verbose=kwargs.get('verbose'), error=True, _fast_abort=True)
                if err < tol:
                    v = head + tail
                    break
                trunc *= 2
                ctx.prec += ctx.prec // 2
                if i == 4:
                    raise ctx.NoConvergence('Euler-Maclaurin summation did not converge')
        finally:
            ctx.prec = prec
        return +v

    def h(*args):
        a_s = list(args[:p])
        b_s = list(args[p:])
        Ts = []
        recz = ctx.one / z
        negz = ctx.fneg(z, exact=True)
        for k in range(q + 1):
            ak = a_s[k]
            C = [negz]
            Cp = [-ak]
            Gn = b_s + [ak] + [a_s[j] - ak for j in range(q + 1) if j != k]
            Gd = a_s + [b_s[j] - ak for j in range(q)]
            Fn = [ak] + [ak - b_s[j] + 1 for j in range(q)]
            Fd = [1 - a_s[j] + ak for j in range(q + 1) if j != k]
            Ts.append((C, Cp, Gn, Gd, Fn, Fd, recz))
        return Ts
    return ctx.hypercomb(h, a_s + b_s, **kwargs)