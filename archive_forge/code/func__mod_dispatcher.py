import functools
from .._utils import set_module
from .numerictypes import (
from .numeric import ndarray, compare_chararrays
from .numeric import array as narray
from numpy.core.multiarray import _vec_string
from numpy.core import overrides
from numpy.compat import asbytes
import numpy
def _mod_dispatcher(a, values):
    return (a, values)