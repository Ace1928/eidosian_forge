import fcntl
import os
from functools import partial
from pyudev._ctypeslib.libc import ERROR_CHECKERS, FD_PAIR, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
def _pipe2_ctypes(libc, flags):
    """A ``pipe2`` implementation using ``pipe2`` from ctypes.

    ``libc`` is a :class:`ctypes.CDLL` object for libc.  ``flags`` is an
    integer providing the flags to ``pipe2``.

    Return a pair of file descriptors ``(r, w)``.

    """
    fds = FD_PAIR()
    libc.pipe2(fds, flags)
    return (fds[0], fds[1])