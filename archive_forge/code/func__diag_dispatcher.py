import functools
import operator
from numpy.core.numeric import (
from numpy.core.overrides import set_array_function_like_doc, set_module
from numpy.core import overrides
from numpy.core import iinfo
from numpy.lib.stride_tricks import broadcast_to
def _diag_dispatcher(v, k=None):
    return (v,)