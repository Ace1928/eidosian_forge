from .functions import defun, defun_wrapped
@defun
def _djacobi_theta3(ctx, z, q, nd):
    """nd=1,2,3 order of the derivative with respect to z"""
    MIN = 2
    extra1 = 10
    extra2 = 20
    if not ctx._im(q) and (not ctx._im(z)):
        s = 0
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = x * x >> wp
        c1, s1 = ctx.cos_sin(ctx._re(z) * 2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if nd & 1:
            s += a * sn >> wp
        else:
            s += a * cn >> wp
        n = 2
        while abs(a) > MIN:
            b = b * x2 >> wp
            a = a * b >> wp
            cn, sn = (cn * c1 - sn * s1 >> wp, sn * c1 + cn * s1 >> wp)
            if nd & 1:
                s += a * sn * n ** nd >> wp
            else:
                s += a * cn * n ** nd >> wp
            n += 1
        s = -(s << nd + 1)
        s = ctx.ldexp(s, -wp)
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = xre * xre - xim * xim >> wp
        x2im = xre * xim >> wp - 1
        are = bre = xre
        aim = bim = xim
        c1, s1 = ctx.cos_sin(ctx._re(z) * 2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if nd & 1:
            sre = are * sn >> wp
            sim = aim * sn >> wp
        else:
            sre = are * cn >> wp
            sim = aim * cn >> wp
        n = 2
        while are ** 2 + aim ** 2 > MIN:
            bre, bim = (bre * x2re - bim * x2im >> wp, bre * x2im + bim * x2re >> wp)
            are, aim = (are * bre - aim * bim >> wp, are * bim + aim * bre >> wp)
            cn, sn = (cn * c1 - sn * s1 >> wp, sn * c1 + cn * s1 >> wp)
            if nd & 1:
                sre += are * sn * n ** nd >> wp
                sim += aim * sn * n ** nd >> wp
            else:
                sre += are * cn * n ** nd >> wp
                sim += aim * cn * n ** nd >> wp
            n += 1
        sre = -(sre << nd + 1)
        sim = -(sim << nd + 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = x * x >> wp
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(2 * z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if nd & 1:
            sre = a * snre >> wp
            sim = a * snim >> wp
        else:
            sre = a * cnre >> wp
            sim = a * cnim >> wp
        n = 2
        while abs(a) > MIN:
            b = b * x2 >> wp
            a = a * b >> wp
            t1 = cnre * c1re - cnim * c1im - snre * s1re + snim * s1im >> wp
            t2 = cnre * c1im + cnim * c1re - snre * s1im - snim * s1re >> wp
            t3 = snre * c1re - snim * c1im + cnre * s1re - cnim * s1im >> wp
            t4 = snre * c1im + snim * c1re + cnre * s1im + cnim * s1re >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if nd & 1:
                sre += a * snre * n ** nd >> wp
                sim += a * snim * n ** nd >> wp
            else:
                sre += a * cnre * n ** nd >> wp
                sim += a * cnim * n ** nd >> wp
            n += 1
        sre = -(sre << nd + 1)
        sim = -(sim << nd + 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = xre * xre - xim * xim >> wp
        x2im = xre * xim >> wp - 1
        are = bre = xre
        aim = bim = xim
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(2 * z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if nd & 1:
            sre = are * snre - aim * snim >> wp
            sim = aim * snre + are * snim >> wp
        else:
            sre = are * cnre - aim * cnim >> wp
            sim = aim * cnre + are * cnim >> wp
        n = 2
        while are ** 2 + aim ** 2 > MIN:
            bre, bim = (bre * x2re - bim * x2im >> wp, bre * x2im + bim * x2re >> wp)
            are, aim = (are * bre - aim * bim >> wp, are * bim + aim * bre >> wp)
            t1 = cnre * c1re - cnim * c1im - snre * s1re + snim * s1im >> wp
            t2 = cnre * c1im + cnim * c1re - snre * s1im - snim * s1re >> wp
            t3 = snre * c1re - snim * c1im + cnre * s1re - cnim * s1im >> wp
            t4 = snre * c1im + snim * c1re + cnre * s1im + cnim * s1re >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if nd & 1:
                sre += (are * snre - aim * snim) * n ** nd >> wp
                sim += (aim * snre + are * snim) * n ** nd >> wp
            else:
                sre += (are * cnre - aim * cnim) * n ** nd >> wp
                sim += (aim * cnre + are * cnim) * n ** nd >> wp
            n += 1
        sre = -(sre << nd + 1)
        sim = -(sim << nd + 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    if nd & 1:
        return (-1) ** (nd // 2) * s
    else:
        return (-1) ** (1 + nd // 2) * s