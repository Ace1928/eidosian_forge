import functools
from .._utils import set_module
from .numerictypes import (
from .numeric import ndarray, compare_chararrays
from .numeric import array as narray
from numpy.core.multiarray import _vec_string
from numpy.core import overrides
from numpy.compat import asbytes
import numpy
def _is_unicode(arr):
    """Returns True if arr is a string or a string array with a dtype that
    represents a unicode string, otherwise returns False.

    """
    if isinstance(arr, str) or issubclass(numpy.asarray(arr).dtype.type, str):
        return True
    return False