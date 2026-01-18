import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _validate_gpass_gstop(gpass, gstop):
    if gpass <= 0.0:
        raise ValueError('gpass should be larger than 0.0')
    elif gstop <= 0.0:
        raise ValueError('gstop should be larger than 0.0')
    elif gpass > gstop:
        raise ValueError('gpass should be smaller than gstop')