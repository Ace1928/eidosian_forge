import warnings
from numbers import Integral
import numpy as np
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _safe_indexing
from ..utils._param_validation import Hidden, Interval, Options, StrOptions
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
from ._encoders import OneHotEncoder
def _validate_n_bins(self, n_features):
    """Returns n_bins_, the number of bins per feature."""
    orig_bins = self.n_bins
    if isinstance(orig_bins, Integral):
        return np.full(n_features, orig_bins, dtype=int)
    n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)
    if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
        raise ValueError('n_bins must be a scalar or array of shape (n_features,).')
    bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)
    violating_indices = np.where(bad_nbins_value)[0]
    if violating_indices.shape[0] > 0:
        indices = ', '.join((str(i) for i in violating_indices))
        raise ValueError('{} received an invalid number of bins at indices {}. Number of bins must be at least 2, and must be an int.'.format(KBinsDiscretizer.__name__, indices))
    return n_bins