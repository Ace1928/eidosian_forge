from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import get_scorer_names
from ..model_selection import check_cv, cross_val_score
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        