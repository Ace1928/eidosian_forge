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
def _use_cache(self, estimator):
    """Return True if using a cache is beneficial, thus when a response method will
        be called several time.
        """
    if len(self._scorers) == 1:
        return False
    counter = Counter([_check_response_method(estimator, scorer._response_method).__name__ for scorer in self._scorers.values() if isinstance(scorer, _BaseScorer)])
    if any((val > 1 for val in counter.values())):
        return True
    return False