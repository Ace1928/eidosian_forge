from array import array
from threading import Lock
import traceback
import ctypes
from ctypes import c_int, c_void_p, CFUNCTYPE, POINTER, cast
from .base import _LIB, check_call, string_types, mx_uint
from .base import NDArrayHandle, c_array, c_handle_array, c_array_buf, MXCallbackList, SymbolHandle
from .ndarray import NDArray, _ndarray_cls
from .ndarray import _GRAD_REQ_MAP
from .symbol import Symbol
class _RecordingStateScope(object):
    """Scope for managing training state.

    Example::

        with _RecordingStateScope(True, True):
            y = model(x)
            backward([y])

    """

    def __init__(self, is_record, train_mode):
        self._enter_is_record = is_record
        self._enter_train_mode = train_mode
        self._prev_is_record = None
        self._prev_train_mode = None

    def __enter__(self):
        if self._enter_is_record is not None:
            self._prev_is_record = set_recording(self._enter_is_record)
        if self._enter_train_mode is not None:
            self._prev_train_mode = set_training(self._enter_train_mode)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_record is not None and self._prev_is_record != self._enter_is_record:
            set_recording(self._prev_is_record)
        if self._enter_train_mode is not None and self._prev_train_mode != self._enter_train_mode:
            set_training(self._prev_train_mode)