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
def _get_vhdx_block_size(self, vhdx_file):
    metadata_offset = self._get_vhdx_metadata_size_and_offset(vhdx_file)[1]
    offset = metadata_offset + VHDX_BS_METADATA_ENTRY_OFFSET
    vhdx_file.seek(offset)
    file_parameter_offset = struct.unpack('<I', vhdx_file.read(4))[0]
    vhdx_file.seek(file_parameter_offset + metadata_offset)
    block_size = struct.unpack('<I', vhdx_file.read(4))[0]
    return block_size