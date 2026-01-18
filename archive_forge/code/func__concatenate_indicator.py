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
def _concatenate_indicator(self, X_imputed, X_indicator):
    """Concatenate indicator mask with the imputed data."""
    if not self.add_indicator:
        return X_imputed
    if sp.issparse(X_imputed):
        hstack = partial(sp.hstack, format=X_imputed.format)
    else:
        hstack = np.hstack
    if X_indicator is None:
        raise ValueError('Data from the missing indicator are not provided. Call _fit_indicator and _transform_indicator in the imputer implementation.')
    return hstack((X_imputed, X_indicator))