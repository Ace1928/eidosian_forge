import inspect
import pprint
from collections import OrderedDict
from .._config import get_config
from ..base import BaseEstimator
from . import is_scalar_nan
def _format_params(self, items, stream, indent, allowance, context, level):
    return self._format_params_or_dict_items(items, stream, indent, allowance, context, level, is_dict=False)