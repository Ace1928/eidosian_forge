from math import log, exp
def _incomplete_gamma(x, alpha):
    """Compute an incomplete gamma ratio (PRIVATE).

    Comments from Z. Yang::

        Returns the incomplete gamma ratio I(x,alpha) where x is the upper
               limit of the integration and alpha is the shape parameter.
        returns (-1) if in error
        ln_gamma_alpha = ln(Gamma(alpha)), is almost redundant.
        (1) series expansion     if alpha>x or x<=1
        (2) continued fraction   otherwise
        RATNEST FORTRAN by
        Bhattacharjee GP (1970) The incomplete gamma integral.  Applied Statistics,
        19: 285-287 (AS32)

    """
    p = alpha
    g = _ln_gamma_function(alpha)
    accurate = 1e-08
    overflow = 1e+30
    gin = 0
    rn = 0
    a = 0
    b = 0
    an = 0
    dif = 0
    term = 0
    if x == 0:
        return 0
    if x < 0 or p <= 0:
        return -1
    factor = exp(p * log(x) - x - g)
    if x > 1 and x >= p:
        a = 1 - p
        b = a + x + 1
        term = 0
        pn = [1, x, x + 1, x * b, None, None]
        gin = pn[2] / pn[3]
    else:
        gin = 1
        term = 1
        rn = p
        while term > accurate:
            rn += 1
            term *= x / rn
            gin += term
        gin *= factor / p
        return gin
    while True:
        a += 1
        b += 2
        term += 1
        an = a * term
        for i in range(2):
            pn[i + 4] = b * pn[i + 2] - an * pn[i]
        if pn[5] != 0:
            rn = pn[4] / pn[5]
            dif = abs(gin - rn)
            if dif > accurate:
                gin = rn
            elif dif <= accurate * rn:
                break
        for i in range(4):
            pn[i] = pn[i + 2]
        if abs(pn[4]) < overflow:
            continue
        for i in range(4):
            pn[i] /= overflow
    gin = 1 - factor * gin
    return gin