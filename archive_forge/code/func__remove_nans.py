import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _remove_nans(samples, paired):
    """Remove nans from paired or unpaired 1D samples"""
    if not paired:
        return [sample[~np.isnan(sample)] for sample in samples]
    nans = np.isnan(samples[0])
    for sample in samples[1:]:
        nans = nans | np.isnan(sample)
    not_nans = ~nans
    return [sample[not_nans] for sample in samples]