import operator
import math
import cmath
def _gamma_complex(x):
    if not x.imag:
        return complex(_gamma_real(x.real))
    if x.real < 0.5:
        return pi / (_sinpi_complex(x) * _gamma_complex(1 - x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g + 2):
            r += _lanczos_p[i] / (x + i)
        t = x + _lanczos_g + 0.5
        return 2.5066282746310007 * t ** (x + 0.5) * cmath.exp(-t) * r