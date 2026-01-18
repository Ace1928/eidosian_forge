from array import array
import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array, c_array_buf, c_handle_array
from ..ndarray import NDArray, zeros_like, _GRAD_REQ_MAP
class TrainingStateScope(object):
    """Scope for managing training state.

    Example::
        with TrainingStateScope(True):
            y = model(x)
            compute_gradient([y])
    """

    def __init__(self, enter_state):
        self._enter_state = enter_state
        self._prev = None

    def __enter__(self):
        self._prev = set_is_training(self._enter_state)

    def __exit__(self, ptype, value, trace):
        if self._prev != self._enter_state:
            set_is_training(self._prev)