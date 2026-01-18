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
def _create_win_event(name):
    from winappdbg.win32.kernel32 import CreateEventA, WaitForSingleObject, CloseHandle
    manual_reset = False
    initial_state = False
    if not isinstance(name, bytes):
        name = name.encode('utf-8')
    event = CreateEventA(None, manual_reset, initial_state, name)
    if not event:
        raise ctypes.WinError()

    class _WinEvent(object):

        def wait_for_event_set(self, timeout=None):
            """
            :param timeout: in seconds
            """
            if timeout is None:
                timeout = 4294967295
            else:
                timeout = int(timeout * 1000)
            ret = WaitForSingleObject(event, timeout)
            if ret in (0, 128):
                return True
            elif ret == 258:
                return False
            else:
                raise ctypes.WinError()
    try:
        yield _WinEvent()
    finally:
        CloseHandle(event)