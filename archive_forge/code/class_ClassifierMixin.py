import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
class ClassifierMixin:
    """Mixin class for all classifiers in scikit-learn.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"classifier"`;
    - `score` method that default to :func:`~sklearn.metrics.accuracy_score`.
    - enforce that `fit` requires `y` to be passed through the `requires_y` tag.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, ClassifierMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(ClassifierMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=1)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([1, 1, 1])
    >>> estimator.score(X, y)
    0.66...
    """
    _estimator_type = 'classifier'

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _more_tags(self):
        return {'requires_y': True}