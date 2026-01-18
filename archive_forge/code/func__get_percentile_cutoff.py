from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def _get_percentile_cutoff(data, cutoff=None, percentile=None, required=False):
    """Get a cutoff for a dataset.

    Parameters
    ----------
    data : array-like
    cutoff : float or None, optional (default: None)
        Absolute cutoff value. Only one of cutoff and percentile may be given
    percentile : float or None, optional (default: None)
        Percentile cutoff value between 0 and 100.
        Only one of cutoff and percentile may be given
    required : bool, optional (default: False)
        If True, one of cutoff and percentile must be given.

    Returns
    -------
    cutoff : float or None
        Absolute cutoff value. Can only be None if required is False and
        cutoff and percentile are both None.
    """
    if percentile is not None:
        if cutoff is not None:
            raise ValueError('Only one of `cutoff` and `percentile` should be given.Got cutoff={}, percentile={}'.format(cutoff, percentile))
        if not isinstance(percentile, numbers.Number):
            return [_get_percentile_cutoff(data, percentile=p) for p in percentile]
        if percentile < 1:
            warnings.warn('`percentile` expects values between 0 and 100.Got {}. Did you mean {}?'.format(percentile, percentile * 100), UserWarning)
        cutoff = np.percentile(np.array(data).reshape(-1), percentile)
    elif cutoff is None and required:
        raise ValueError('One of either `cutoff` or `percentile` must be given.')
    return cutoff