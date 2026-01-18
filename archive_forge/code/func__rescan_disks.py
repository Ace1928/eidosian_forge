import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
@_utils.retry_decorator(exceptions=(exceptions.x_wmi, exceptions.OSWinException))
def _rescan_disks(self):
    LOG.debug('Rescanning disks.')
    ret = self._conn_storage.Msft_StorageSetting.UpdateHostStorageCache()
    if isinstance(ret, Iterable):
        ret = ret[0]
    if ret:
        err_msg = _('Rescanning disks failed. Error code: %s.')
        raise exceptions.OSWinException(err_msg % ret)
    LOG.debug('Finished rescanning disks.')