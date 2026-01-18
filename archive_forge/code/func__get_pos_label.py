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
def _get_pos_label(self):
    if 'pos_label' in self._kwargs:
        return self._kwargs['pos_label']
    score_func_params = signature(self._score_func).parameters
    if 'pos_label' in score_func_params:
        return score_func_params['pos_label'].default
    return None