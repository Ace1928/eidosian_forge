from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun
def _hurwitz(ctx, s, a=1, d=0, **kwargs):
    prec = ctx.prec
    verbose = kwargs.get('verbose')
    try:
        extraprec = 10
        ctx.prec += extraprec
        a, atype = ctx._convert_param(a)
        if ctx.re(s) < 0:
            if verbose:
                print('zeta: Attempting reflection formula')
            try:
                return _hurwitz_reflection(ctx, s, a, d, atype)
            except NotImplementedError:
                pass
            if verbose:
                print('zeta: Reflection formula failed')
        if verbose:
            print('zeta: Using the Euler-Maclaurin algorithm')
        while 1:
            ctx.prec = prec + extraprec
            T1, T2 = _hurwitz_em(ctx, s, a, d, prec + 10, verbose)
            cancellation = ctx.mag(T1) - ctx.mag(T1 + T2)
            if verbose:
                print('Term 1:', T1)
                print('Term 2:', T2)
                print('Cancellation:', cancellation, 'bits')
            if cancellation < extraprec:
                return T1 + T2
            else:
                extraprec = max(2 * extraprec, min(cancellation + 5, 100 * prec))
                if extraprec > kwargs.get('maxprec', 100 * prec):
                    raise ctx.NoConvergence('zeta: too much cancellation')
    finally:
        ctx.prec = prec