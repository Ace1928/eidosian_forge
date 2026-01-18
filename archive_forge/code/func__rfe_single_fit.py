from numbers import Integral
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import check_scoring
from ..model_selection import check_cv
from ..model_selection._validation import _score
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split, available_if
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin, _get_feature_importances
def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    return rfe._fit(X_train, y_train, lambda estimator, features: _score(estimator, X_test[:, features], y_test, scorer, score_params=None)).scores_