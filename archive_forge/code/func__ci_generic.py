from .functions import defun, defun_wrapped
@defun_wrapped
def _ci_generic(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf:
            return ctx.zero
        if z == ctx.ninf:
            return ctx.pi * 1j
    jz = ctx.fmul(ctx.j, z, exact=True)
    njz = ctx.fneg(jz, exact=True)
    v = 0.5 * (ctx.ei(jz) + ctx.ei(njz))
    zreal = ctx._re(z)
    zimag = ctx._im(z)
    if zreal == 0:
        if zimag > 0:
            v += ctx.pi * 0.5j
        if zimag < 0:
            v -= ctx.pi * 0.5j
    if zreal < 0:
        if zimag >= 0:
            v += ctx.pi * 1j
        if zimag < 0:
            v -= ctx.pi * 1j
    if ctx._is_real_type(z) and zreal > 0:
        v = ctx._re(v)
    return v