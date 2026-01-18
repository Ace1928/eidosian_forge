import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _find_nat_freq(stopb, passb, gpass, gstop, filter_type, filter_kind):
    if filter_type == 1:
        nat = stopb / passb
    elif filter_type == 2:
        nat = passb / stopb
    elif filter_type == 3:
        wp0 = _optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12, args=(0, passb, stopb, gpass, gstop, filter_kind), disp=0)
        passb[0] = wp0
        wp1 = _optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1], args=(1, passb, stopb, gpass, gstop, filter_kind), disp=0)
        passb[1] = wp1
        nat = stopb * (passb[0] - passb[1]) / (stopb ** 2 - passb[0] * passb[1])
    elif filter_type == 4:
        nat = (stopb ** 2 - passb[0] * passb[1]) / (stopb * (passb[0] - passb[1]))
    else:
        raise ValueError(f'should not happen: filter_type ={filter_type!r}.')
    nat = min(cupy.abs(nat))
    return (nat, passb)