import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _trim_zeros(filt, trim='fb'):
    first = 0
    if 'f' in trim:
        for i in filt:
            if i != 0.0:
                break
            else:
                first = first + 1
    last = len(filt)
    if 'b' in trim:
        for i in filt[::-1]:
            if i != 0.0:
                break
            else:
                last = last - 1
    return filt[first:last]