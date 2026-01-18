import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import (  # type: ignore
from ..utils._metadata_requests import (
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _preprocess_data
def _estimate_noise_variance(self, X, y, positive):
    """Compute an estimate of the variance with an OLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to be fitted by the OLS model. We expect the data to be
            centered.

        y : ndarray of shape (n_samples,)
            Associated target.

        positive : bool, default=False
            Restrict coefficients to be >= 0. This should be inline with
            the `positive` parameter from `LassoLarsIC`.

        Returns
        -------
        noise_variance : float
            An estimator of the noise variance of an OLS model.
        """
    if X.shape[0] <= X.shape[1] + self.fit_intercept:
        raise ValueError(f'You are using {self.__class__.__name__} in the case where the number of samples is smaller than the number of features. In this setting, getting a good estimate for the variance of the noise is not possible. Provide an estimate of the noise variance in the constructor.')
    ols_model = LinearRegression(positive=positive, fit_intercept=False)
    y_pred = ols_model.fit(X, y).predict(X)
    return np.sum((y - y_pred) ** 2) / (X.shape[0] - X.shape[1] - self.fit_intercept)