from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun
def _zetasum(ctx, s, a, n, derivatives=[0], reflect=False):
    """
    Returns [xd0,xd1,...,xdr], [yd0,yd1,...ydr] where

    xdk = D^k     ( 1/a^s     +  1/(a+1)^s      +  ...  +  1/(a+n)^s     )
    ydk = D^k conj( 1/a^(1-s) +  1/(a+1)^(1-s)  +  ...  +  1/(a+n)^(1-s) )

    D^k = kth derivative with respect to s, k ranges over the given list of
    derivatives (which should consist of either a single element
    or a range 0,1,...r). If reflect=False, the ydks are not computed.
    """
    if abs(ctx.re(s)) < 0.5 * ctx.prec:
        try:
            return ctx._zetasum_fast(s, a, n, derivatives, reflect)
        except NotImplementedError:
            pass
    negs = ctx.fneg(s, exact=True)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1
    if not reflect:
        if not have_derivatives:
            return ([ctx.fsum(((a + k) ** negs for k in xrange(n + 1)))], [])
        if have_one_derivative:
            d = derivatives[0]
            x = ctx.fsum((ctx.ln(a + k) ** d * (a + k) ** negs for k in xrange(n + 1)))
            return ([(-1) ** d * x], [])
    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd + 1)
    xs = [ctx.zero for d in derivatives]
    if reflect:
        ys = [ctx.zero for d in derivatives]
    else:
        ys = []
    for k in xrange(n + 1):
        w = a + k
        xterm = w ** negs
        if reflect:
            yterm = ctx.conj(ctx.one / (w * xterm))
        if have_derivatives:
            logw = -ctx.ln(w)
            if have_one_derivative:
                logw = logw ** maxd
                xs[0] += xterm * logw
                if reflect:
                    ys[0] += yterm * logw
            else:
                t = ctx.one
                for d in derivatives:
                    xs[d] += xterm * t
                    if reflect:
                        ys[d] += yterm * t
                    t *= logw
        else:
            xs[0] += xterm
            if reflect:
                ys[0] += yterm
    return (xs, ys)