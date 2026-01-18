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
@staticmethod
def _basic_indexing_slice_is_contiguous(slc_key, shape):
    """Whether indexing with the given key results in a contiguous array.

        The rule is: From right to left, if in an axis, a slice produces a
        proper subset, the later slice must have <=1 elements.

        The ``slc_key`` sequence must have the same length as ``shape`` and
        only contain `slice` objects.
        """
    assert len(slc_key) == len(shape)
    is_subset = False
    total_sliced_elements = np.prod([_get_slice_len(slc, n) for slc, n in zip(slc_key, shape)])
    if total_sliced_elements in (0, 1):
        return True
    for idx, n in zip(reversed(slc_key), reversed(shape)):
        _, _, step = idx.indices(n)
        num_elements = _get_slice_len(idx, n)
        if num_elements == 0:
            return True
        elif num_elements > 1 and (step > 1 or step < 0):
            return False
        elif is_subset:
            if num_elements > 1:
                return False
        elif num_elements < n:
            is_subset = True
    return True