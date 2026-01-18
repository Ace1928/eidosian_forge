import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _single_zpksos(z, p, k):
    """Create one second-order section from up to two zeros and poles"""
    sos = cupy.zeros(6)
    b, a = zpk2tf(cupy.asarray(z), cupy.asarray(p), k)
    sos[3 - len(b):3] = b
    sos[6 - len(a):6] = a
    return sos