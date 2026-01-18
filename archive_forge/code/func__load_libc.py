from __future__ import with_statement
import os
import errno
import struct
import threading
import ctypes
import ctypes.util
from functools import reduce
from ctypes import c_int, c_char_p, c_uint32
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import UnsupportedLibc
def _load_libc():
    libc_path = None
    try:
        libc_path = ctypes.util.find_library('c')
    except (OSError, IOError, RuntimeError):
        pass
    if libc_path is not None:
        return ctypes.CDLL(libc_path)
    try:
        return ctypes.CDLL('libc.so')
    except (OSError, IOError):
        pass
    try:
        return ctypes.CDLL('libc.so.6')
    except (OSError, IOError):
        pass
    try:
        return ctypes.CDLL('libc.so.0')
    except (OSError, IOError) as err:
        raise err