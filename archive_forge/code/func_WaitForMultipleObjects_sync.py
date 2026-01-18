from __future__ import annotations
import math
import trio
from ._core._windows_cffi import (
def WaitForMultipleObjects_sync(*handles: int | CData) -> None:
    """Wait for any of the given Windows handles to be signaled."""
    n = len(handles)
    handle_arr = handle_array(n)
    for i in range(n):
        handle_arr[i] = handles[i]
    timeout = 4294967295
    retcode = kernel32.WaitForMultipleObjects(n, handle_arr, False, timeout)
    if retcode == ErrorCodes.WAIT_FAILED:
        raise_winerror()