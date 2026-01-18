import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _postprocess_wn(WN, analog, fs):
    wn = WN if analog else cupy.arctan(WN) * 2.0 / pi
    if len(wn) == 1:
        wn = wn[0]
    if fs is not None:
        wn = wn * fs / 2
    return wn