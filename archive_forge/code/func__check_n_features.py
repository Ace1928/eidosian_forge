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
def _check_n_features(self, X, reset):
    """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
    try:
        n_features = _num_features(X)
    except TypeError as e:
        if not reset and hasattr(self, 'n_features_in_'):
            raise ValueError(f'X does not contain any features, but {self.__class__.__name__} is expecting {self.n_features_in_} features') from e
        return
    if reset:
        self.n_features_in_ = n_features
        return
    if not hasattr(self, 'n_features_in_'):
        return
    if n_features != self.n_features_in_:
        raise ValueError(f'X has {n_features} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input.')