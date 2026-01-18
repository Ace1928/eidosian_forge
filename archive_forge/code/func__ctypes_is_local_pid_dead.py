import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def _ctypes_is_local_pid_dead(pid):
    """True if pid doesn't correspond to live process on this machine"""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(1, False, pid)
    if not handle:
        errorcode = ctypes.GetLastError()
        if errorcode == 5:
            return False
        elif errorcode == 87:
            return True
        raise ctypes.WinError(errorcode)
    kernel32.CloseHandle(handle)
    return False