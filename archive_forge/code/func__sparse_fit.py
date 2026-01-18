import numbers
import warnings
from collections import Counter
from functools import partial
import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _is_pandas_na, is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import MissingValues, StrOptions
from ..utils.fixes import _mode
from ..utils.sparsefuncs import _get_median
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
def _sparse_fit(self, X, strategy, missing_values, fill_value):
    """Fit the transformer on sparse data."""
    missing_mask = _get_mask(X, missing_values)
    mask_data = missing_mask.data
    n_implicit_zeros = X.shape[0] - np.diff(X.indptr)
    statistics = np.empty(X.shape[1])
    if strategy == 'constant':
        statistics.fill(fill_value)
    else:
        for i in range(X.shape[1]):
            column = X.data[X.indptr[i]:X.indptr[i + 1]]
            mask_column = mask_data[X.indptr[i]:X.indptr[i + 1]]
            column = column[~mask_column]
            mask_zeros = _get_mask(column, 0)
            column = column[~mask_zeros]
            n_explicit_zeros = mask_zeros.sum()
            n_zeros = n_implicit_zeros[i] + n_explicit_zeros
            if len(column) == 0 and self.keep_empty_features:
                statistics[i] = 0
            elif strategy == 'mean':
                s = column.size + n_zeros
                statistics[i] = np.nan if s == 0 else column.sum() / s
            elif strategy == 'median':
                statistics[i] = _get_median(column, n_zeros)
            elif strategy == 'most_frequent':
                statistics[i] = _most_frequent(column, 0, n_zeros)
    super()._fit_indicator(missing_mask)
    return statistics