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
@contextlib.contextmanager
def _get_hba_handle(self, adapter_name=None, adapter_wwn_struct=None):
    if adapter_name:
        hba_handle = self._open_adapter_by_name(adapter_name)
    elif adapter_wwn_struct:
        hba_handle = self._open_adapter_by_wwn(adapter_wwn_struct)
    else:
        err_msg = _('Could not open HBA adapter. No HBA name or WWN was specified')
        raise exceptions.FCException(err_msg)
    try:
        yield hba_handle
    finally:
        self._close_adapter(hba_handle)