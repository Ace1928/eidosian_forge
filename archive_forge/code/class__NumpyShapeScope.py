import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
class _NumpyShapeScope(object):
    """Scope for managing NumPy shape semantics.
    In NumPy shape semantics, `()` represents the shape of scalar tensors,
    and tuples with `0` elements, for example, `(0,)`, `(1, 0, 2)`, represent
    the shapes of zero-size tensors.

    Do not use this class directly. Use `np_shape(active)` instead.

    Example::

        with _NumpyShapeScope(True):
            y = model(x)
            backward([y])

    """

    def __init__(self, is_np_shape):
        self._enter_is_np_shape = is_np_shape
        self._prev_is_np_shape = None

    def __enter__(self):
        if self._enter_is_np_shape is not None:
            self._prev_is_np_shape = set_np_shape(self._enter_is_np_shape)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_np_shape is not None and self._prev_is_np_shape != self._enter_is_np_shape:
            set_np_shape(self._prev_is_np_shape)