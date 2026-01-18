import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted regression values.
        