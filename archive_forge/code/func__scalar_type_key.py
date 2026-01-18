import numbers
import warnings
from .multiarray import (
from .._utils import set_module
from ._string_helpers import (
from ._type_aliases import (
from ._dtype import _kind_name
from builtins import bool, int, float, complex, object, str, bytes
from numpy.compat import long, unicode
def _scalar_type_key(typ):
    """A ``key`` function for `sorted`."""
    dt = dtype(typ)
    return (dt.kind.lower(), dt.itemsize)