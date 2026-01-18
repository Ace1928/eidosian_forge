import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@retry_decorator(error_codes=(w_const.ISDSC_SESSION_BUSY, w_const.ISDSC_DEVICE_BUSY_ON_SESSION))
@ensure_buff_and_retrieve_items(struct_type=iscsi_struct.ISCSI_DEVICE_ON_SESSION, func_requests_buff_sz=False)
def _get_iscsi_session_devices(self, session_id, buff=None, buff_size=None, element_count=None):
    self._run_and_check_output(iscsidsc.GetDevicesForIScsiSessionW, ctypes.byref(session_id), ctypes.byref(element_count), buff)