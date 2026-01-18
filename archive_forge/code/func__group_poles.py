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
def _group_poles(poles, tol, rtype):
    if rtype in ['max', 'maximum']:
        reduce = np.max
    elif rtype in ['min', 'minimum']:
        reduce = np.min
    elif rtype in ['avg', 'mean']:
        reduce = np.mean
    else:
        raise ValueError("`rtype` must be one of {'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")
    unique = []
    multiplicity = []
    pole = poles[0]
    block = [pole]
    for i in range(1, len(poles)):
        if abs(poles[i] - pole) <= tol:
            block.append(pole)
        else:
            unique.append(reduce(block))
            multiplicity.append(len(block))
            pole = poles[i]
            block = [pole]
    unique.append(reduce(block))
    multiplicity.append(len(block))
    return (np.asarray(unique), np.asarray(multiplicity))