from . import measure
from . import utils
from scipy import sparse
from sklearn.preprocessing import normalize
import numbers
import numpy as np
import pandas as pd
import warnings
def _get_scaled_libsize(data, rescale=10000, return_library_size=False):
    if return_library_size or rescale in ['median', 'mean']:
        libsize = measure.library_size(data)
    else:
        libsize = None
    if rescale == 'median':
        rescale = np.median(utils.toarray(libsize))
        if rescale == 0:
            warnings.warn('Median library size is zero. Rescaling to mean instead.', UserWarning)
            rescale = np.mean(utils.toarray(libsize))
    elif rescale == 'mean':
        rescale = np.mean(utils.toarray(libsize))
    elif isinstance(rescale, numbers.Number):
        pass
    elif rescale is None:
        rescale = 1
    else:
        raise ValueError("Expected rescale in ['median', 'mean'], a number or `None`. Got {}".format(rescale))
    return (rescale, libsize)