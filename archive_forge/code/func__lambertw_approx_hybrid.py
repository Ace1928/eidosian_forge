from ..libmp.backend import xrange
import math
import cmath
def _lambertw_approx_hybrid(z, k):
    imag_sign = 0
    if hasattr(z, 'imag'):
        x = float(z.real)
        y = z.imag
        if y:
            imag_sign = (-1) ** (y < 0)
        y = float(y)
    else:
        x = float(z)
        y = 0.0
        imag_sign = 0
    if not y:
        y = 0.0
    z = complex(x, y)
    if k == 0:
        if -4.0 < y < 4.0 and -1.0 < x < 2.5:
            if imag_sign:
                if y > 1.0:
                    return 0.876 + 0.645j + (0.118 - 0.174j) * (z - (0.75 + 2.5j))
                if y > 0.25:
                    return 0.505 + 0.204j + (0.375 - 0.132j) * (z - (0.75 + 0.5j))
                if y < -1.0:
                    return 0.876 - 0.645j + (0.118 + 0.174j) * (z - (0.75 - 2.5j))
                if y < -0.25:
                    return 0.505 - 0.204j + (0.375 + 0.132j) * (z - (0.75 - 0.5j))
            if x < -0.5:
                if imag_sign >= 0:
                    return -0.318 + 1.34j + (-0.697 - 0.593j) * (z + 1)
                else:
                    return -0.318 - 1.34j + (-0.697 + 0.593j) * (z + 1)
            r = -0.367879441171442
            if not imag_sign and x > r:
                z = x
            if x < -0.2:
                return -1 + 2.33164398159712 * (z - r) ** 0.5 - 1.81218788563936 * (z - r)
            if x < 0.5:
                return z
            return 0.2 + 0.3 * z
        if not imag_sign and x > 0.0:
            L1 = math.log(x)
            L2 = math.log(L1)
        else:
            L1 = cmath.log(z)
            L2 = cmath.log(L1)
    elif k == -1:
        r = -0.367879441171442
        if not imag_sign and r < x < 0.0:
            z = x
        if imag_sign >= 0 and y < 0.1 and (-0.6 < x < -0.2):
            return -1 - 2.33164398159712 * (z - r) ** 0.5 - 1.81218788563936 * (z - r)
        if not imag_sign and -0.2 <= x < 0.0:
            L1 = math.log(-x)
            return L1 - math.log(-L1)
        else:
            if imag_sign == -1 and (not y) and (x < 0.0):
                L1 = cmath.log(z) - 3.141592653589793j
            else:
                L1 = cmath.log(z) - 6.283185307179586j
            L2 = cmath.log(L1)
    return L1 - L2 + L2 / L1 + L2 * (L2 - 2) / (2 * L1 ** 2)