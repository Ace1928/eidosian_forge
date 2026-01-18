from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def _scatter_set_nd(self, value_nd, indices):
    """
        This is added as an NDArray class method in order to support polymorphism in NDArray and numpy.ndarray indexing
        """
    return _internal._scatter_set_nd(lhs=self, rhs=value_nd, indices=indices, shape=self.shape, out=self)