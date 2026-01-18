import ctypes
import os
import struct
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import virtdisk as vdisk_struct
from os_win.utils.winapi import wintypes
def _get_vhd_info_member(self, vhd_file, info_member):
    virt_disk_info = vdisk_struct.GET_VIRTUAL_DISK_INFO()
    virt_disk_info.Version = ctypes.c_uint(info_member)
    infoSize = ctypes.sizeof(virt_disk_info)
    virtdisk.GetVirtualDiskInformation.restype = wintypes.DWORD
    ignored_error_codes = []
    if info_member == w_const.GET_VIRTUAL_DISK_INFO_PARENT_LOCATION:
        ignored_error_codes.append(w_const.ERROR_VHD_INVALID_TYPE)
    self._run_and_check_output(virtdisk.GetVirtualDiskInformation, vhd_file, ctypes.byref(ctypes.c_ulong(infoSize)), ctypes.byref(virt_disk_info), None, ignored_error_codes=ignored_error_codes)
    return self._parse_vhd_info(virt_disk_info, info_member)