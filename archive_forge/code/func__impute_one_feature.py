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
def _impute_one_feature(self, X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, estimator=None, fit_mode=True):
    """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using
        the current estimates of all the other features. The `estimator` must
        support `return_std=True` in its `predict` method for this function
        to work.

        Parameters
        ----------
        X_filled : ndarray
            Input data with the most recent imputations.

        mask_missing_values : ndarray
            Input data's missing indicator matrix.

        feat_idx : int
            Index of the feature currently being imputed.

        neighbor_feat_idx : ndarray
            Indices of the features to be used in imputing `feat_idx`.

        estimator : object
            The estimator to use at this step of the round-robin imputation.
            If `sample_posterior=True`, the estimator must support
            `return_std` in its `predict` method.
            If None, it will be cloned from self._estimator.

        fit_mode : boolean, default=True
            Whether to fit and predict with the estimator or just predict.

        Returns
        -------
        X_filled : ndarray
            Input data with `X_filled[missing_row_mask, feat_idx]` updated.

        estimator : estimator with sklearn API
            The fitted estimator used to impute
            `X_filled[missing_row_mask, feat_idx]`.
        """
    if estimator is None and fit_mode is False:
        raise ValueError('If fit_mode is False, then an already-fitted estimator should be passed in.')
    if estimator is None:
        estimator = clone(self._estimator)
    missing_row_mask = mask_missing_values[:, feat_idx]
    if fit_mode:
        X_train = _safe_indexing(_safe_indexing(X_filled, neighbor_feat_idx, axis=1), ~missing_row_mask, axis=0)
        y_train = _safe_indexing(_safe_indexing(X_filled, feat_idx, axis=1), ~missing_row_mask, axis=0)
        estimator.fit(X_train, y_train)
    if np.sum(missing_row_mask) == 0:
        return (X_filled, estimator)
    X_test = _safe_indexing(_safe_indexing(X_filled, neighbor_feat_idx, axis=1), missing_row_mask, axis=0)
    if self.sample_posterior:
        mus, sigmas = estimator.predict(X_test, return_std=True)
        imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
        positive_sigmas = sigmas > 0
        imputed_values[~positive_sigmas] = mus[~positive_sigmas]
        mus_too_low = mus < self._min_value[feat_idx]
        imputed_values[mus_too_low] = self._min_value[feat_idx]
        mus_too_high = mus > self._max_value[feat_idx]
        imputed_values[mus_too_high] = self._max_value[feat_idx]
        inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
        mus = mus[inrange_mask]
        sigmas = sigmas[inrange_mask]
        a = (self._min_value[feat_idx] - mus) / sigmas
        b = (self._max_value[feat_idx] - mus) / sigmas
        truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
        imputed_values[inrange_mask] = truncated_normal.rvs(random_state=self.random_state_)
    else:
        imputed_values = estimator.predict(X_test)
        imputed_values = np.clip(imputed_values, self._min_value[feat_idx], self._max_value[feat_idx])
    _safe_assign(X_filled, imputed_values, row_indexer=missing_row_mask, column_indexer=feat_idx)
    return (X_filled, estimator)