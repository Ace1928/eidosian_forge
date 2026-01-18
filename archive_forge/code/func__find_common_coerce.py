import numbers
import warnings
from .multiarray import (
from .._utils import set_module
from ._string_helpers import (
from ._type_aliases import (
from ._dtype import _kind_name
from builtins import bool, int, float, complex, object, str, bytes
from numpy.compat import long, unicode
def _find_common_coerce(a, b):
    if a > b:
        return a
    try:
        thisind = __test_types.index(a.char)
    except ValueError:
        return None
    return _can_coerce_all([a, b], start=thisind)