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
def _get_internal_vhd_size_by_file_size(self, vhd_path, new_vhd_file_size, vhd_info):
    """Fixed VHD size = Data Block size + 512 bytes

           | Dynamic_VHD_size = Dynamic Disk Header
           |                  + Copy of hard disk footer
           |                  + Hard Disk Footer
           |                  + Data Block
           |                  + BAT
           | Dynamic Disk header fields
           |     Copy of hard disk footer (512 bytes)
           |     Dynamic Disk Header (1024 bytes)
           |     BAT (Block Allocation table)
           |     Data Block 1
           |     Data Block 2
           |     Data Block n
           |     Hard Disk Footer (512 bytes)
           | Default block size is 2M
           | BAT entry size is 4byte
        """
    vhd_type = vhd_info['ProviderSubtype']
    if vhd_type == constants.VHD_TYPE_FIXED:
        vhd_header_size = VHD_HEADER_SIZE_FIX
        return new_vhd_file_size - vhd_header_size
    else:
        bs = vhd_info['BlockSize']
        bes = VHD_BAT_ENTRY_SIZE
        ddhs = VHD_DYNAMIC_DISK_HEADER_SIZE
        hs = VHD_HEADER_SIZE_DYNAMIC
        fs = VHD_FOOTER_SIZE_DYNAMIC
        max_internal_size = (new_vhd_file_size - (hs + ddhs + fs)) * bs // (bes + bs)
        return max_internal_size