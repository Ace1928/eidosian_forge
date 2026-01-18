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
def _get_vhdx_log_size(self, vhdx_file):
    current_header_offset = self._get_vhdx_current_header_offset(vhdx_file)
    offset = current_header_offset + VHDX_LOG_LENGTH_OFFSET
    vhdx_file.seek(offset)
    log_size = struct.unpack('<I', vhdx_file.read(4))[0]
    return log_size