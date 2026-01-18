import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def _order_complex_poles(poles):
    """
    Check we have complex conjugates pairs and reorder P according to YT, ie
    real_poles, complex_i, conjugate complex_i, ....
    The lexicographic sort on the complex poles is added to help the user to
    compare sets of poles.
    """
    ordered_poles = cupy.sort(poles[cupy.isreal(poles)])
    im_poles = []
    for p in cupy.sort(poles[cupy.imag(poles) < 0]):
        if cupy.conj(p) in poles:
            im_poles.extend((p, cupy.conj(p)))
    ordered_poles = cupy.hstack((ordered_poles, im_poles))
    if poles.shape[0] != len(ordered_poles):
        raise ValueError('Complex poles must come with their conjugates')
    return ordered_poles