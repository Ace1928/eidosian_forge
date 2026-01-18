import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def _set_np_array(active):
    """Turns on/off NumPy array semantics for the current thread in which `mxnet.numpy.ndarray`
    is expected to be created, instead of the legacy `mx.nd.NDArray`.

    Parameters
    ---------
    active : bool
        A boolean value indicating whether the NumPy-array semantics should be turned on or off.

    Returns
    -------
        A bool value indicating the previous state of NumPy array semantics.
    """
    global _set_np_array_logged
    if active:
        if not _set_np_array_logged:
            import logging
            logging.info('NumPy array semantics has been activated in your code. This allows you to use operators from MXNet NumPy and NumPy Extension modules as well as MXNet NumPy `ndarray`s.')
            _set_np_array_logged = True
    cur_state = is_np_array()
    _NumpyArrayScope._current.value = _NumpyArrayScope(active)
    return cur_state