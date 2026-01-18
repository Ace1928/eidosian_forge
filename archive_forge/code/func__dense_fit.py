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
def _dense_fit(self, X, strategy, missing_values, fill_value):
    """Fit the transformer on dense data."""
    missing_mask = _get_mask(X, missing_values)
    masked_X = ma.masked_array(X, mask=missing_mask)
    super()._fit_indicator(missing_mask)
    if strategy == 'mean':
        mean_masked = np.ma.mean(masked_X, axis=0)
        mean = np.ma.getdata(mean_masked)
        mean[np.ma.getmask(mean_masked)] = 0 if self.keep_empty_features else np.nan
        return mean
    elif strategy == 'median':
        median_masked = np.ma.median(masked_X, axis=0)
        median = np.ma.getdata(median_masked)
        median[np.ma.getmaskarray(median_masked)] = 0 if self.keep_empty_features else np.nan
        return median
    elif strategy == 'most_frequent':
        X = X.transpose()
        mask = missing_mask.transpose()
        if X.dtype.kind == 'O':
            most_frequent = np.empty(X.shape[0], dtype=object)
        else:
            most_frequent = np.empty(X.shape[0])
        for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
            row_mask = np.logical_not(row_mask).astype(bool)
            row = row[row_mask]
            if len(row) == 0 and self.keep_empty_features:
                most_frequent[i] = 0
            else:
                most_frequent[i] = _most_frequent(row, np.nan, 0)
        return most_frequent
    elif strategy == 'constant':
        return np.full(X.shape[1], fill_value, dtype=X.dtype)