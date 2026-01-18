import inspect
import pprint
from collections import OrderedDict
from .._config import get_config
from ..base import BaseEstimator
from . import is_scalar_nan
def _pprint_estimator(self, object, stream, indent, allowance, context, level):
    stream.write(object.__class__.__name__ + '(')
    if self._indent_at_name:
        indent += len(object.__class__.__name__)
    if self._changed_only:
        params = _changed_params(object)
    else:
        params = object.get_params(deep=False)
    params = OrderedDict(((name, val) for name, val in sorted(params.items())))
    self._format_params(params.items(), stream, indent, allowance + 1, context, level)
    stream.write(')')