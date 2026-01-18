import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
def _get_response_method(response_method, needs_threshold, needs_proba):
    """Handles deprecation of `needs_threshold` and `needs_proba` parameters in
    favor of `response_method`.
    """
    needs_threshold_provided = needs_threshold != 'deprecated'
    needs_proba_provided = needs_proba != 'deprecated'
    response_method_provided = response_method is not None
    needs_threshold = False if needs_threshold == 'deprecated' else needs_threshold
    needs_proba = False if needs_proba == 'deprecated' else needs_proba
    if response_method_provided and (needs_proba_provided or needs_threshold_provided):
        raise ValueError('You cannot set both `response_method` and `needs_proba` or `needs_threshold` at the same time. Only use `response_method` since the other two are deprecated in version 1.4 and will be removed in 1.6.')
    if needs_proba_provided or needs_threshold_provided:
        warnings.warn('The `needs_threshold` and `needs_proba` parameter are deprecated in version 1.4 and will be removed in 1.6. You can either let `response_method` be `None` or set it to `predict` to preserve the same behaviour.', FutureWarning)
    if response_method_provided:
        return response_method
    if needs_proba is True and needs_threshold is True:
        raise ValueError('You cannot set both `needs_proba` and `needs_threshold` at the same time. Use `response_method` instead since the other two are deprecated in version 1.4 and will be removed in 1.6.')
    if needs_proba is True:
        response_method = 'predict_proba'
    elif needs_threshold is True:
        response_method = ('decision_function', 'predict_proba')
    else:
        response_method = 'predict'
    return response_method