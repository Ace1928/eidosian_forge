from numbers import Integral, Real
import numpy as np
from scipy import optimize
from ..base import BaseEstimator, RegressorMixin, _fit_context
from ..utils import axis0_safe_slice
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.optimize import _check_optimize_result
from ..utils.validation import _check_sample_weight
from ._base import LinearModel
Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,)
            Weight given to each sample.

        Returns
        -------
        self : object
            Fitted `HuberRegressor` estimator.
        