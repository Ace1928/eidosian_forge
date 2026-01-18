from .functions import defun, defun_wrapped
def _hermite_param(ctx, n, z, parabolic_cylinder):
    """
    Combined calculation of the Hermite polynomial H_n(z) (and its
    generalization to complex n) and the parabolic cylinder
    function D.
    """
    n, ntyp = ctx._convert_param(n)
    z = ctx.convert(z)
    q = -ctx.mpq_1_2
    if not z:
        T1 = ([2, ctx.pi], [n, 0.5], [], [q * (n - 1)], [], [], 0)
        if parabolic_cylinder:
            T1[1][0] += q * n
        return (T1,)
    can_use_2f0 = ctx.isnpint(-n) or ctx.re(z) > 0 or (ctx.re(z) == 0 and ctx.im(z) > 0)
    expprec = ctx.prec * 4 + 20
    if parabolic_cylinder:
        u = ctx.fmul(ctx.fmul(z, z, prec=expprec), -0.25, exact=True)
        w = ctx.fmul(z, ctx.sqrt(0.5, prec=expprec), prec=expprec)
    else:
        w = z
    w2 = ctx.fmul(w, w, prec=expprec)
    rw2 = ctx.fdiv(1, w2, prec=expprec)
    nrw2 = ctx.fneg(rw2, exact=True)
    nw = ctx.fneg(w, exact=True)
    if can_use_2f0:
        T1 = ([2, w], [n, n], [], [], [q * n, q * (n - 1)], [], nrw2)
        terms = [T1]
    else:
        T1 = ([2, nw], [n, n], [], [], [q * n, q * (n - 1)], [], nrw2)
        T2 = ([2, ctx.pi, nw], [n + 2, 0.5, 1], [], [q * n], [q * (n - 1)], [1 - q], w2)
        terms = [T1, T2]
    if parabolic_cylinder:
        expu = ctx.exp(u)
        for i in range(len(terms)):
            terms[i][1][0] += q * n
            terms[i][0].append(expu)
            terms[i][1].append(1)
    return tuple(terms)