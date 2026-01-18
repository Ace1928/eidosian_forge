from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def _information_gain(error_histogram):
    """
    Helper function to calculate the information gain of the given error
    histogram.

    Parameters
    ----------
    error_histogram : numpy array
        Error histogram.

    Returns
    -------
    information_gain : float
        Information gain.

    """
    if np.asarray(error_histogram).any():
        entropy = _entropy(error_histogram)
    else:
        entropy = 0.0
    return np.log2(len(error_histogram)) - entropy