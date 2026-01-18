from .functions import defun, defun_wrapped
@defun_wrapped
def _erf_complex(ctx, z):
    z2 = ctx.square_exp_arg(z, -1)
    v = 2 / ctx.sqrt(ctx.pi) * z * ctx.hyp1f1((1, 2), (3, 2), z2)
    if not ctx._re(z):
        v = ctx._im(v) * ctx.j
    return v