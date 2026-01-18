from .polyerrors import HeuristicGCDFailed
def _gcd_interpolate(h, x, ring):
    """Interpolate polynomial GCD from integer GCD. """
    f, i = (ring.zero, 0)
    if ring.ngens == 1:
        while h:
            g = h % x
            if g > x // 2:
                g -= x
            h = (h - g) // x
            if g:
                f[i,] = g
            i += 1
    else:
        while h:
            g = h.trunc_ground(x)
            h = (h - g).quo_ground(x)
            if g:
                for monom, coeff in g.iterterms():
                    f[(i,) + monom] = coeff
            i += 1
    if f.LC < 0:
        return -f
    else:
        return f