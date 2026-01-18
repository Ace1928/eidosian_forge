from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def _histogram_bins(num_bins):
    """
    Helper function to generate the histogram bins used to calculate the error
    histogram of the information gain.

    Parameters
    ----------
    num_bins : int
        Number of histogram bins.
    Returns
    -------
    numpy array
        Histogram bin edges.

    Notes
    -----
    This functions returns the bin edges for a histogram with one more bin than
    the requested number of bins, because the fist and last bins are added
    together (to make the histogram circular) later on. Because of the same
    reason, the first and the last bin are only half as wide as the others.

    """
    if num_bins % 2 != 0 or num_bins < 2:
        raise ValueError('Number of error histogram bins must be even and greater than 0')
    offset = 0.5 / num_bins
    return np.linspace(-0.5 - offset, 0.5 + offset, num_bins + 2)