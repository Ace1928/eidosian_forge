import math
from .functions import defun
def Rzeta_set(ctx, s, derivatives=[0]):
    """
    Computes several derivatives of the auxiliary function of Riemann `R(s)`.

    **Definition**

    The function is defined by

    .. math ::

        \\begin{equation}
        {\\mathop{\\mathcal R }\\nolimits}(s)=
        \\int_{0\\swarrow1}\\frac{x^{-s} e^{\\pi i x^2}}{e^{\\pi i x}-
        e^{-\\pi i x}}\\,dx
        \\end{equation}

    To this function we apply the Riemann-Siegel expansion.
    """
    der = max(derivatives)
    wpinitial = ctx.prec
    t = ctx._im(s)
    sigma = ctx._re(s)
    ctx.prec = 15
    a = ctx.sqrt(t / (2 * ctx.pi))
    asigma = ctx.power(a, sigma)
    A1 = ctx.power(2, ctx.mag(asigma) - 1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps / 6.0
    eps2 = eps * A1 / 3.0
    ctx.prec = 15
    if sigma > 0:
        b = 2.0
        c = math.pow(9, sigma) / 4.44288
        A = math.pow(9, sigma)
        B1 = 1
    else:
        b = 2.25158
        c = math.pow(2, -sigma) / 4.44288
        A = math.pow(2, -sigma)
        B1 = 1.10789
    ctx.prec = 15
    L = 1
    while 3 * c * ctx.gamma(L * 0.5) * ctx.power(b * a, -L) >= eps2:
        L = L + 1
    L = max(2, L)
    if 3 * L >= 2 * a * a / 25.0 or 3 * L + 2 + sigma < 0 or abs(sigma) > a / 2.0:
        ctx.prec = wpinitial
        raise NotImplementedError('Riemann-Siegel can not compute with such precision')
    eps3 = eps2 / (4 * L)
    eps4 = eps3 / (3 * L)
    M = aux_M_Fp(ctx, A, eps4, a, B1, L)
    Fp = {}
    for n in range(M, 3 * L - 2):
        Fp[n] = 0
    h1 = eps4 / (632 * A)
    h2 = ctx.pi * ctx.pi * B1 * a * ctx.sqrt(3) * math.e * math.e
    h2 = h1 * ctx.power(h2 / M ** 2, (M - 1) / 3) / M
    h3 = min(h1, h2)
    J = 12
    jvalue = (2 * ctx.pi) ** J / ctx.gamma(J + 1)
    while jvalue > h3:
        J = J + 1
        jvalue = 2 * ctx.pi * jvalue / J
    eps5 = {}
    foreps5 = math.pi * math.pi * B1 * a
    for m in range(0, 22):
        aux1 = math.pow(foreps5, m / 3) / (316.0 * A)
        aux2 = ctx.gamma(m + 1) / ctx.gamma(m / 3.0 + 0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = aux1 * aux2 * eps4
    twenty = min(3 * L - 3, 21) + 1
    aux = 6812 * J
    wpfp = ctx.mag(44 * J)
    for m in range(0, twenty):
        wpfp = max(wpfp, ctx.mag(aux * ctx.gamma(m + 1) / eps5[m]))
    ctx.prec = wpfp + ctx.mag(t) + 20
    a = ctx.sqrt(t / (2 * ctx.pi))
    N = ctx.floor(a)
    p = 1 - 2 * (a - N)
    num = ctx.floor(p * ctx.mpf(2) ** wpfp)
    difference = p * ctx.mpf(2) ** wpfp - num
    if difference < 0.5:
        num = num
    else:
        num = num + 1
    p = ctx.convert(num * ctx.mpf(2) ** (-wpfp))
    eps6 = ctx.power(2 * ctx.pi, J) / (ctx.gamma(J + 1) * 3 * J)
    cc = {}
    cont = {}
    cont, pipowers = coef(ctx, J, eps6)
    cc = cont.copy()
    Fp = {}
    for n in range(M, 3 * L - 2):
        Fp[n] = 0
    ctx.prec = wpfp
    for m in range(0, M + 1):
        sumP = 0
        for k in range(2 * J - m - 1, -1, -1):
            sumP = sumP * p + cc[k]
        Fp[m] = sumP
        for k in range(0, 2 * J - m - 1):
            cc[k] = (k + 1) * cc[k + 1]
    wpd = {}
    d1 = max(6, ctx.mag(40 * L * L))
    d2 = 13 + ctx.mag((1 + abs(sigma)) * A) - ctx.mag(eps4) - 1
    const = ctx.ln(8 / (ctx.pi * ctx.pi * a * a * B1 * B1)) / 2
    for n in range(0, L):
        d3 = ctx.mag(ctx.sqrt(ctx.gamma(n - 0.5))) - ctx.floor(n * const) + d2
        wpd[n] = max(d3, d1)
    ctx.prec = wpd[1] + 10
    psigma = 1 - 2 * sigma
    d = {}
    d[0, 0, -2] = 0
    d[0, 0, -1] = 0
    d[0, 0, 0] = 1
    d[0, 0, 1] = 0
    d[0, -1, -2] = 0
    d[0, -1, -1] = 0
    d[0, -1, 0] = 1
    d[0, -1, 1] = 0
    for n in range(1, L):
        ctx.prec = wpd[n] + 10
        for k in range(0, 3 * n // 2 + 1):
            m = 3 * n - 2 * k
            if m != 0:
                m1 = ctx.one / m
                c1 = m1 / 4
                c2 = psigma * m1 / 2
                c3 = -(m + 1)
                d[0, n, k] = c3 * d[0, n - 1, k - 2] + c1 * d[0, n - 1, k] + c2 * d[0, n - 1, k - 1]
            else:
                d[0, n, k] = 0
                for r in range(0, k):
                    add = d[0, n, r] * (ctx.one * ctx.fac(2 * k - 2 * r) / ctx.fac(k - r))
                    d[0, n, k] -= (-1) ** (k - r) * add
        d[0, n, -2] = 0
        d[0, n, -1] = 0
        d[0, n, 3 * n // 2 + 1] = 0
    for mu in range(-2, der + 1):
        for n in range(-2, L):
            for k in range(-3, max(1, 3 * n // 2 + 2)):
                if mu < 0 or n < 0 or k < 0 or (k > 3 * n // 2):
                    d[mu, n, k] = 0
    for mu in range(1, der + 1):
        for n in range(0, L):
            ctx.prec = wpd[n] + 10
            for k in range(0, 3 * n // 2 + 1):
                aux = (2 * mu - 2) * d[mu - 2, n - 2, k - 3] + 2 * (sigma + n - 2) * d[mu - 1, n - 2, k - 3]
                d[mu, n, k] = aux - d[mu - 1, n - 1, k - 1]
    wptcoef = {}
    wpterm = {}
    ctx.prec = 15
    c1 = ctx.mag(40 * (L + 2))
    c2 = ctx.mag(68 * (L + 2) * A)
    c4 = ctx.mag(B1 * a * math.sqrt(ctx.pi)) - 1
    for k in range(0, L):
        c3 = c2 - k * c4 + ctx.mag(ctx.fac(k + 0.5)) / 2.0
        wptcoef[k] = max(c1, c3 - ctx.mag(eps4) + 1) + 1 + 10
        wpterm[k] = max(c1, ctx.mag(L + 2) + c3 - ctx.mag(eps3) + 1) + 1 + 10
    fortcoef = {}
    for mu in derivatives:
        for k in range(0, L):
            for ell in range(-2, 3 * k // 2 + 1):
                fortcoef[mu, k, ell] = 0
    for mu in derivatives:
        for k in range(0, L):
            ctx.prec = wptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                fortcoef[mu, k, ell] = d[mu, k, ell] * Fp[3 * k - 2 * ell] / pipowers[2 * k - ell]
                fortcoef[mu, k, ell] = fortcoef[mu, k, ell] / (2 * ctx.j) ** ell

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t / (2 * ctx.pi))
        ctx.prec = wp
        return aa
    tcoef = {}
    for chi in derivatives:
        for k in range(0, L):
            for ell in range(-2, 3 * k // 2 + 1):
                tcoef[chi, k, ell] = 0
    ctx.prec = wptcoef[0] + 3
    aa = trunc_a(t)
    la = -ctx.ln(aa)
    for chi in derivatives:
        for k in range(0, L):
            ctx.prec = wptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                tcoef[chi, k, ell] = 0
                for mu in range(0, chi + 1):
                    tcoefter = ctx.binomial(chi, mu) * la ** mu * fortcoef[chi - mu, k, ell]
                    tcoef[chi, k, ell] += tcoefter
    ctx.prec = wptcoef[0] + 2
    av = {}
    av[0] = 1
    av[1] = av[0] / a
    ctx.prec = wptcoef[0]
    for k in range(2, L):
        av[k] = av[k - 1] * av[1]
    tv = {}
    for chi in derivatives:
        for k in range(0, L):
            ctx.prec = wptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                tv[chi, k, ell] = tcoef[chi, k, ell] * av[k]
    term = {}
    for chi in derivatives:
        for n in range(0, L):
            ctx.prec = wpterm[n]
            te = 0
            for k in range(0, 3 * n // 2 + 1):
                te += tv[chi, n, k]
            term[chi, n] = te
    rssum = {}
    ctx.prec = 15
    rsbound = math.sqrt(ctx.pi) * c / (b * a)
    ctx.prec = 15
    wprssum = ctx.mag(4.4 * (L + 3) ** 2 * rsbound / eps2)
    wprssum = max(wprssum, ctx.mag(10 * (L + 1)))
    ctx.prec = wprssum
    for chi in derivatives:
        rssum[chi] = 0
        for k in range(1, L + 1):
            rssum[chi] += term[chi, L - k]
    ctx.prec = 15
    A2 = 2 ** ctx.mag(rssum[0])
    eps8 = eps / (3 * A2)
    T = t * ctx.ln(t / (2 * ctx.pi))
    wps3 = 5 + ctx.mag((1 + 2 / eps8 * ctx.power(a, -sigma)) * T)
    ctx.prec = wps3
    tpi = t / (2 * ctx.pi)
    arg = t / 2 * ctx.ln(tpi) - t / 2 - ctx.pi / 8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    asigma = ctx.power(a, -sigma)
    S3 = (-1) ** (N - 1) * asigma * U
    ctx.prec = 15
    wpsum = 4 + ctx.mag((N + ctx.power(N, 1 - sigma)) * ctx.ln(N) / eps1)
    ctx.prec = wpsum + 10
    '\n    # This can be improved\n    S1 = {}\n    for chi in derivatives:\n        S1[chi] = 0\n    for n in range(1,int(N)+1):\n        ln = ctx.ln(n)\n        expn = ctx.exp(-ln*(sigma+ctx.j*t))\n        for chi in derivatives:\n            term = ctx.power(-ln, chi)*expn\n            S1[chi] += term\n    '
    S1 = ctx._zetasum(s, 1, int(N) - 1, derivatives)[0]
    ctx.prec = 15
    absS1 = abs(S1[der])
    absS2 = abs(rssum[der] * S3)
    wpend = max(6, wpinitial + ctx.mag(6 * (3 * absS1 + 7 * absS2)))
    ctx.prec = wpend
    rz = {}
    for chi in derivatives:
        rz[chi] = S1[chi] + rssum[chi] * S3
    ctx.prec = wpinitial
    return rz