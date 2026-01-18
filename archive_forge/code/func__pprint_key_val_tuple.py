import inspect
import pprint
from collections import OrderedDict
from .._config import get_config
from ..base import BaseEstimator
from . import is_scalar_nan
def _pprint_key_val_tuple(self, object, stream, indent, allowance, context, level):
    """Pretty printing for key-value tuples from dict or parameters."""
    k, v = object
    rep = self._repr(k, context, level)
    if isinstance(object, KeyValTupleParam):
        rep = rep.strip("'")
        middle = '='
    else:
        middle = ': '
    stream.write(rep)
    stream.write(middle)
    self._format(v, stream, indent + len(rep) + len(middle), allowance, context, level)