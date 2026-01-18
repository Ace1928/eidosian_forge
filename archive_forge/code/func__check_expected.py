import contextlib
import ctypes
import os
from ctypes.wintypes import (
from shellingham._core import SHELL_NAMES
def _check_expected(expected):

    def check(ret, func, args):
        if ret:
            return True
        code = ctypes.GetLastError()
        if code == expected:
            return False
        raise ctypes.WinError(code)
    return check