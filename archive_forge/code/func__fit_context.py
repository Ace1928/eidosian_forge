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
def _fit_context(*, prefer_skip_nested_validation):
    """Decorator to run the fit methods of estimators within context managers.

    Parameters
    ----------
    prefer_skip_nested_validation : bool
        If True, the validation of parameters of inner estimators or functions
        called during fit will be skipped.

        This is useful to avoid validating many times the parameters passed by the
        user from the public facing API. It's also useful to avoid validating
        parameters that we pass internally to inner functions that are guaranteed to
        be valid by the test suite.

        It should be set to True for most estimators, except for those that receive
        non-validated objects as parameters, such as meta-estimators that are given
        estimator objects.

    Returns
    -------
    decorated_fit : method
        The decorated fit method.
    """

    def decorator(fit_method):

        @functools.wraps(fit_method)
        def wrapper(estimator, *args, **kwargs):
            global_skip_validation = get_config()['skip_parameter_validation']
            partial_fit_and_fitted = fit_method.__name__ == 'partial_fit' and _is_fitted(estimator)
            if not global_skip_validation and (not partial_fit_and_fitted):
                estimator._validate_params()
            with config_context(skip_parameter_validation=prefer_skip_nested_validation or global_skip_validation):
                return fit_method(estimator, *args, **kwargs)
        return wrapper
    return decorator