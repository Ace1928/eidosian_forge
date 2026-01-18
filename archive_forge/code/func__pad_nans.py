import numpy as np
import scipy.fftpack as fft
from scipy import signal
from statsmodels.tools.validation import array_like, PandasWrapper
def _pad_nans(x, head=None, tail=None):
    if np.ndim(x) == 1:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[np.nan] * head, x, [np.nan] * tail]
        elif tail is None:
            return np.r_[[np.nan] * head, x]
        elif head is None:
            return np.r_[x, [np.nan] * tail]
    elif np.ndim(x) == 2:
        if head is None and tail is None:
            return x
        elif head and tail:
            return np.r_[[[np.nan] * x.shape[1]] * head, x, [[np.nan] * x.shape[1]] * tail]
        elif tail is None:
            return np.r_[[[np.nan] * x.shape[1]] * head, x]
        elif head is None:
            return np.r_[x, [[np.nan] * x.shape[1]] * tail]
    else:
        raise ValueError('Nan-padding for ndim > 2 not implemented')