from .functions import defun, defun_wrapped
@defun_wrapped
def _si_generic(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf:
            return 0.5 * ctx.pi
        if z == ctx.ninf:
            return -0.5 * ctx.pi
    if ctx.mag(z) >= -1:
        jz = ctx.fmul(ctx.j, z, exact=True)
        njz = ctx.fneg(jz, exact=True)
        v = -0.5j * (ctx.ei(jz) - ctx.ei(njz))
        zreal = ctx._re(z)
        if zreal > 0:
            v -= 0.5 * ctx.pi
        if zreal < 0:
            v += 0.5 * ctx.pi
        if ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    else:
        return z * ctx.hyp1f2((1, 2), (3, 2), (3, 2), -0.25 * z * z)