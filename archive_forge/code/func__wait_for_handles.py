from __future__ import unicode_literals
from ..terminal.win32_input import ConsoleInputReader
from ..win32_types import SECURITY_ATTRIBUTES
from .base import EventLoop, INPUT_TIMEOUT
from .inputhook import InputHookContext
from .utils import TimeIt
from ctypes import windll, pointer
from ctypes.wintypes import DWORD, BOOL, HANDLE
import msvcrt
import threading
def _wait_for_handles(handles, timeout=-1):
    """
    Waits for multiple handles. (Similar to 'select') Returns the handle which is ready.
    Returns `None` on timeout.

    http://msdn.microsoft.com/en-us/library/windows/desktop/ms687025(v=vs.85).aspx
    """
    arrtype = HANDLE * len(handles)
    handle_array = arrtype(*handles)
    ret = windll.kernel32.WaitForMultipleObjects(len(handle_array), handle_array, BOOL(False), DWORD(timeout))
    if ret == WAIT_TIMEOUT:
        return None
    else:
        h = handle_array[ret]
        return h