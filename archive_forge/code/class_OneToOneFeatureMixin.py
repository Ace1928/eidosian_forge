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
class OneToOneFeatureMixin:
    """Provides `get_feature_names_out` for simple transformers.

    This mixin assumes there's a 1-to-1 correspondence between input features
    and output features, such as :class:`~sklearn.preprocessing.StandardScaler`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import OneToOneFeatureMixin
    >>> class MyEstimator(OneToOneFeatureMixin):
    ...     def fit(self, X, y=None):
    ...         self.n_features_in_ = X.shape[1]
    ...         return self
    >>> X = np.array([[1, 2], [3, 4]])
    >>> MyEstimator().fit(X).get_feature_names_out()
    array(['x0', 'x1'], dtype=object)
    """

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input features.
        """
        check_is_fitted(self, 'n_features_in_')
        return _check_feature_names_in(self, input_features)