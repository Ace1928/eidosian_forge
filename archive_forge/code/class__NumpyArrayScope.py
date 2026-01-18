import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
class _NumpyArrayScope(object):
    """Scope for managing NumPy array creation. This is often used
    with `is_np_array=True` in initializer to enforce array creation
    as type `mxnet.numpy.ndarray`, instead of `mx.nd.NDArray` in Gluon.

    Do not use this class directly. Use `np_array(active)` instead.
    """
    _current = threading.local()

    def __init__(self, is_np_array):
        self._old_scope = None
        self._is_np_array = is_np_array

    def __enter__(self):
        if not hasattr(_NumpyArrayScope._current, 'value'):
            _NumpyArrayScope._current.value = _NumpyArrayScope(False)
        self._old_scope = _NumpyArrayScope._current.value
        _NumpyArrayScope._current.value = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        _NumpyArrayScope._current.value = self._old_scope