import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _check_custom_lims(custom_lims, x_min, x_max):
    """Check if `custom_lims` are of the correct type.

    It accepts numeric lists/tuples of length 2.

    Parameters
    ----------
    custom_lims : Object whose type is checked.

    Returns
    -------
    None: Object of type None
    """
    if not isinstance(custom_lims, (list, tuple)):
        raise TypeError(f'`custom_lims` must be a numeric list or tuple of length 2.\nNot an object of {type(custom_lims)}.')
    if len(custom_lims) != 2:
        raise AttributeError(f'`len(custom_lims)` must be 2, not {len(custom_lims)}.')
    any_bool = any((isinstance(i, bool) for i in custom_lims))
    if any_bool:
        raise TypeError('Elements of `custom_lims` must be numeric or None, not bool.')
    custom_lims = list(custom_lims)
    if custom_lims[0] is None:
        custom_lims[0] = x_min
    if custom_lims[1] is None:
        custom_lims[1] = x_max
    all_numeric = all((isinstance(i, (int, float, np.integer, np.number)) for i in custom_lims))
    if not all_numeric:
        raise TypeError('Elements of `custom_lims` must be numeric or None.\nAt least one of them is not.')
    if not custom_lims[0] < custom_lims[1]:
        raise ValueError('`custom_lims[0]` must be smaller than `custom_lims[1]`.')
    if custom_lims[0] > x_min or custom_lims[1] < x_max:
        raise ValueError('Some observations are outside `custom_lims` boundaries.')
    return custom_lims