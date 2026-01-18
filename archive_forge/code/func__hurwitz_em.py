from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def _hurwitz_em(ctx, s, a, d, prec, verbose):
    a = ctx.convert(a)
    tol = -prec
    M1 = 0
    M2 = prec // 3
    N = M2
    lsum = 0
    if ctx.isint(s):
        s = int(ctx._re(s))
    s1 = s - 1
    while 1:
        l = ctx._zetasum(s, M1 + a, M2 - M1 - 1, [d])[0][0]
        lsum += l
        M2a = M2 + a
        logM2a = ctx.ln(M2a)
        logM2ad = logM2a ** d
        logs = [logM2ad]
        logr = 1 / logM2a
        rM2a = 1 / M2a
        M2as = M2a ** (-s)
        if d:
            tailsum = ctx.gammainc(d + 1, s1 * logM2a) / s1 ** (d + 1)
        else:
            tailsum = 1 / (s1 * M2a ** s1)
        tailsum += 0.5 * logM2ad * M2as
        U = [1]
        r = M2as
        fact = 2
        for j in range(1, N + 1):
            j2 = 2 * j
            if j == 1:
                upds = [1]
            else:
                upds = [j2 - 2, j2 - 1]
            for m in upds:
                D = min(m, d + 1)
                if m <= d:
                    logs.append(logs[-1] * logr)
                Un = [0] * (D + 1)
                for i in xrange(D):
                    Un[i] = (1 - m - s) * U[i]
                for i in xrange(1, D + 1):
                    Un[i] += (d - (i - 1)) * U[i - 1]
                U = Un
                r *= rM2a
            t = ctx.fdot(U, logs) * r * ctx.bernoulli(j2) / -fact
            tailsum += t
            if ctx.mag(t) < tol:
                return (lsum, (-1) ** d * tailsum)
            fact *= (j2 + 1) * (j2 + 2)
        if verbose:
            print('Sum range:', M1, M2, 'term magnitude', ctx.mag(t), 'tolerance', tol)
        M1, M2 = (M2, M2 * 2)
        if ctx.re(s) < 0:
            N += N // 2