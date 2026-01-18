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
def _parse_vhd_info(self, virt_disk_info, info_member):
    vhd_info = {}
    vhd_info_member = self._vhd_info_members[info_member]
    info = getattr(virt_disk_info, vhd_info_member)
    if hasattr(info, '_fields_'):
        for field in info._fields_:
            vhd_info[field[0]] = getattr(info, field[0])
    else:
        vhd_info[vhd_info_member] = info
    return vhd_info