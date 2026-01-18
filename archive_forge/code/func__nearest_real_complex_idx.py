import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _nearest_real_complex_idx(fro, to, which):
    """Get the next closest real or complex element based on distance"""
    assert which in ('real', 'complex', 'any')
    order = cupy.argsort(cupy.abs(fro - to))
    if which == 'any':
        return order[0]
    else:
        mask = cupy.isreal(fro[order])
        if which == 'complex':
            mask = ~mask
        return order[cupy.nonzero(mask)[0][0]]