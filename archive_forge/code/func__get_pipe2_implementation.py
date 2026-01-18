import fcntl
import os
from functools import partial
from pyudev._ctypeslib.libc import ERROR_CHECKERS, FD_PAIR, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
def _get_pipe2_implementation():
    """
    Find the appropriate implementation for ``pipe2``.

    Return a function implementing ``pipe2``."""
    if hasattr(os, 'pipe2'):
        return os.pipe2
    try:
        libc = load_ctypes_library('libc', SIGNATURES, ERROR_CHECKERS)
        return partial(_pipe2_ctypes, libc) if hasattr(libc, 'pipe2') else _pipe2_by_pipe
    except ImportError:
        return _pipe2_by_pipe