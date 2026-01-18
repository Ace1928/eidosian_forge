import operator
import math
import cmath
def _digamma_complex(x):
    if not x.imag:
        return complex(_digamma_real(x.real))
    if x.real < 0.5:
        x = 1.0 - x
        s = pi * cotpi(x)
    else:
        s = 0.0
    while abs(x) < 10.0:
        s -= 1.0 / x
        x += 1.0
    x2 = x ** (-2)
    t = x2
    for c in _psi_coeff:
        s -= c * t
        if abs(t) < 1e-20:
            break
        t *= x2
    return s + cmath.log(x) - 0.5 / x