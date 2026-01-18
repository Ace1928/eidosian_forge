import warnings
from collections import namedtuple
from numbers import Integral, Real
from time import time
import numpy as np
from scipy import stats
from ..base import _fit_context, clone
from ..exceptions import ConvergenceWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from ._base import SimpleImputer, _BaseImputer, _check_inputs_dtype
def _initial_imputation(self, X, in_fit=False):
    """Perform initial imputation for input `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        in_fit : bool, default=False
            Whether function is called in :meth:`fit`.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        X_filled : ndarray of shape (n_samples, n_features)
            Input data with the most recent imputations.

        mask_missing_values : ndarray of shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features,
            masked by non-missing features.

        X_missing_mask : ndarray, shape (n_samples, n_features)
            Input data's mask matrix indicating missing datapoints, where
            `n_samples` is the number of samples and `n_features` is the
            number of features.
        """
    if is_scalar_nan(self.missing_values):
        force_all_finite = 'allow-nan'
    else:
        force_all_finite = True
    X = self._validate_data(X, dtype=FLOAT_DTYPES, order='F', reset=in_fit, force_all_finite=force_all_finite)
    _check_inputs_dtype(X, self.missing_values)
    X_missing_mask = _get_mask(X, self.missing_values)
    mask_missing_values = X_missing_mask.copy()
    if self.initial_imputer_ is None:
        self.initial_imputer_ = SimpleImputer(missing_values=self.missing_values, strategy=self.initial_strategy, fill_value=self.fill_value, keep_empty_features=self.keep_empty_features).set_output(transform='default')
        X_filled = self.initial_imputer_.fit_transform(X)
    else:
        X_filled = self.initial_imputer_.transform(X)
    valid_mask = np.flatnonzero(np.logical_not(np.isnan(self.initial_imputer_.statistics_)))
    if not self.keep_empty_features:
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]
    else:
        mask_missing_values[:, valid_mask] = True
        Xt = X
    return (Xt, X_filled, mask_missing_values, X_missing_mask)