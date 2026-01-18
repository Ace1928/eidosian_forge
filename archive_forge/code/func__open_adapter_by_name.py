import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def _open_adapter_by_name(self, adapter_name):
    handle = self._run_and_check_output(hbaapi.HBA_OpenAdapter, ctypes.c_char_p(six.b(adapter_name)), ret_val_is_err_code=False, error_on_nonzero_ret_val=False, error_ret_vals=[0])
    return handle