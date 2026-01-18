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
def _warn_overlap(self, message, kwargs):
    """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.

        This method is intended to be used to check for overlap between
        ``self._kwargs`` and ``kwargs`` passed as metadata.
        """
    _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
    overlap = _kwargs.intersection(kwargs.keys())
    if overlap:
        warnings.warn(f'{message} Overlapping parameters are: {overlap}', UserWarning)