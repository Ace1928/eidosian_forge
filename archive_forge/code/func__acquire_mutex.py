import ctypes
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
import platform
import traceback
import os, time, sys
@contextmanager
def _acquire_mutex(mutex_name, timeout):
    """
    Only one process may be attaching to a pid, so, create a system mutex
    to make sure this holds in practice.
    """
    from winappdbg.win32.kernel32 import CreateMutex, GetLastError, CloseHandle
    from winappdbg.win32.defines import ERROR_ALREADY_EXISTS
    initial_time = time.time()
    while True:
        mutex = CreateMutex(None, True, mutex_name)
        acquired = GetLastError() != ERROR_ALREADY_EXISTS
        if acquired:
            break
        if time.time() - initial_time > timeout:
            raise TimeoutError('Unable to acquire mutex to make attach before timeout.')
        time.sleep(0.2)
    try:
        yield
    finally:
        CloseHandle(mutex)