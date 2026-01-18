from .functions import defun, defun_wrapped
@defun_wrapped
def _erfc_complex(ctx, z):
    if ctx.re(z) > 2:
        z2 = ctx.square_exp_arg(z)
        nz2 = ctx.fneg(z2, exact=True)
        v = ctx.exp(nz2) / ctx.sqrt(ctx.pi) * ctx.hyperu((1, 2), (1, 2), z2)
    else:
        v = 1 - ctx._erf_complex(z)
    if not ctx._re(z):
        v = 1 + ctx._im(v) * ctx.j
    return v