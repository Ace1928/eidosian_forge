from __future__ import annotations
import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
import pytest
import sniffio
from .. import (
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
from ..testing import wait_all_tasks_blocked
def _get_thread_name(ident: int | None=None) -> str | None:
    import ctypes
    import ctypes.util
    libpthread_path = ctypes.util.find_library('pthread')
    if not libpthread_path:
        libpthread_path = 'libc.so'
    try:
        libpthread = ctypes.CDLL(libpthread_path)
    except Exception:
        print(f'no pthread on {sys.platform}')
        return None
    pthread_getname_np = getattr(libpthread, 'pthread_getname_np', None)
    assert pthread_getname_np
    pthread_getname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    pthread_getname_np.restype = ctypes.c_int
    name_buffer = ctypes.create_string_buffer(b'', size=16)
    if ident is None:
        ident = threading.get_ident()
    assert pthread_getname_np(ident, name_buffer, 16) == 0
    try:
        return name_buffer.value.decode()
    except UnicodeDecodeError as e:
        pytest.fail(f'value: {name_buffer.value!r}, exception: {e}')