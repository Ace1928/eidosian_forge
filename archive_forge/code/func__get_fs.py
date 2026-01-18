from math import ceil, log
import operator
import warnings
import numpy as np
from numpy.fft import irfft, fft, ifft
from scipy.special import sinc
from scipy.linalg import (toeplitz, hankel, solve, LinAlgError, LinAlgWarning,
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from . import _sigtools
def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    if nyq is _NoValue and fs is None:
        fs = 2
    elif nyq is not _NoValue:
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        msg = "Keyword argument 'nyq' is deprecated in favour of 'fs' and will be removed in SciPy 1.14.0."
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        if nyq is None:
            fs = 2
        else:
            fs = 2 * nyq
    return fs