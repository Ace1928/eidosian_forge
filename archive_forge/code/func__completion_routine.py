import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def _completion_routine(error_code, num_bytes, lpOverLapped):
    """Sets the completion event and executes callback, if passed."""
    overlapped = ctypes.cast(lpOverLapped, wintypes.LPOVERLAPPED).contents
    self.set_event(overlapped.hEvent)
    if callback:
        callback(num_bytes)