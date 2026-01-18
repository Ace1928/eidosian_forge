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
def _get_disk_by_number(self, disk_number, msft_disk_cls=True):
    if msft_disk_cls:
        disk = self._conn_storage.Msft_Disk(Number=disk_number)
    else:
        disk = self._conn_cimv2.Win32_DiskDrive(Index=disk_number)
    if not disk:
        err_msg = _('Could not find the disk number %s')
        raise exceptions.DiskNotFound(err_msg % disk_number)
    return disk[0]