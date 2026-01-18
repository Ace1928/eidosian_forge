import operator
import math
from math import prod as _prod
import timeit
import warnings
from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt
def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError("Unknown value '%s' given to padtype.  padtype must be 'even', 'odd', 'constant', or None." % padtype)
    if padtype is None:
        padlen = 0
    if padlen is None:
        edge = ntaps * 3
    else:
        edge = padlen
    if x.shape[axis] <= edge:
        raise ValueError('The length of the input vector x must be greater than padlen, which is %d.' % edge)
    if padtype is not None and edge > 0:
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x
    return (edge, ext)