import errno
import functools
import fcntl
import os
import struct
import threading
from . import exceptions
from . import _error_translation as errors
from .bindings import libzfs_core
from ._constants import MAXNAMELEN
from .ctypes import int32_t
from ._nvlist import nvlist_in, nvlist_out
def _uncommitted_decorator(func, depends_on=depends_on):

    @functools.wraps(func)
    def _f(*args, **kwargs):
        if not is_supported(_f):
            raise NotImplementedError(func.__name__)
        return func(*args, **kwargs)
    if depends_on is not None:
        _f._check_func = depends_on
    return _f