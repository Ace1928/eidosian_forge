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
class _BaseImputer(TransformerMixin, BaseEstimator):
    """Base class for all imputers.

    It adds automatically support for `add_indicator`.
    """
    _parameter_constraints: dict = {'missing_values': [MissingValues()], 'add_indicator': ['boolean'], 'keep_empty_features': ['boolean']}

    def __init__(self, *, missing_values=np.nan, add_indicator=False, keep_empty_features=False):
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _fit_indicator(self, X):
        """Fit a MissingIndicator."""
        if self.add_indicator:
            self.indicator_ = MissingIndicator(missing_values=self.missing_values, error_on_new=False)
            self.indicator_._fit(X, precomputed=True)
        else:
            self.indicator_ = None

    def _transform_indicator(self, X):
        """Compute the indicator mask.'

        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
        if self.add_indicator:
            if not hasattr(self, 'indicator_'):
                raise ValueError('Make sure to call _fit_indicator before _transform_indicator')
            return self.indicator_.transform(X)

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

    def _concatenate_indicator_feature_names_out(self, names, input_features):
        if not self.add_indicator:
            return names
        indicator_names = self.indicator_.get_feature_names_out(input_features)
        return np.concatenate([names, indicator_names])

    def _more_tags(self):
        return {'allow_nan': is_scalar_nan(self.missing_values)}