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
def _parse_head(heads, head_grads):
    """parse head gradient for backward and grad."""
    if isinstance(heads, NDArray):
        heads = [heads]
    if isinstance(head_grads, NDArray):
        head_grads = [head_grads]
    head_handles = c_handle_array(heads)
    if head_grads is None:
        hgrad_handles = ctypes.c_void_p(0)
    else:
        assert len(heads) == len(head_grads), 'heads and head_grads must be lists of the same length'
        hgrad_handles = c_array(NDArrayHandle, [i.handle if i is not None else NDArrayHandle(0) for i in head_grads])
    return (head_handles, hgrad_handles)