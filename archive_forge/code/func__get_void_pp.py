import ctypes
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
@staticmethod
def _get_void_pp():
    return ctypes.pointer(ctypes.c_void_p())