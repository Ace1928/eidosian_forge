import math
import warnings
from numbers import Real
import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
from .base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from .utils import check_array, check_consistent_length
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.validation import _check_sample_weight, check_is_fitted
def _build_f(self, X, y):
    """Build the f_ interp1d function."""
    bounds_error = self.out_of_bounds == 'raise'
    if len(y) == 1:
        self.f_ = lambda x: y.repeat(x.shape)
    else:
        self.f_ = interpolate.interp1d(X, y, kind='linear', bounds_error=bounds_error)