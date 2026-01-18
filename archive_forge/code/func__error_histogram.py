from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def _error_histogram(detections, annotations, histogram_bins):
    """
    Helper function to calculate the relative errors of the given detections
    and annotations and map them to an histogram with the given bins edges.

    Parameters
    ----------
    detections : list or numpy array
        Detected beats.
    annotations : list or numpy array
        Annotated beats.
    histogram_bins : numpy array
        Beat error histogram bin edges.

    Returns
    -------
    error_histogram : numpy array
        Beat error histogram.

    Notes
    -----
    The returned error histogram is circular, i.e. it contains 1 bin less than
    a histogram built normally with the given histogram bin edges. The values
    of the last and first bin are summed and mapped to the first bin.

    """
    errors = calc_relative_errors(detections, annotations)
    errors = np.mod(errors + 0.5, -1) + 0.5
    histogram = np.histogram(errors, histogram_bins)[0].astype(np.float)
    histogram[0] += histogram[-1]
    return histogram[:-1]