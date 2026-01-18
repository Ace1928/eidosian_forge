import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _transform_integrals(a, b):
    negative = b < a
    a[negative], b[negative] = (b[negative], a[negative])
    abinf = np.isinf(a) & np.isinf(b)
    a[abinf], b[abinf] = (-1, 1)
    ainf = np.isinf(a)
    a[ainf], b[ainf] = (-b[ainf], -a[ainf])
    binf = np.isinf(b)
    a0 = a.copy()
    a[binf], b[binf] = (0, 1)
    return (a, b, a0, negative, abinf, ainf, binf)