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
def _get_internal_vhdx_size_by_file_size(self, vhd_path, new_vhd_file_size, vhd_info):
    """VHDX Size:

        Header (1MB) + Log + Metadata Region + BAT + Payload Blocks

        The chunk size is the maximum number of bytes described by a SB
        block.

        Chunk size = 2^{23} * SectorSize

        :param str vhd_path: VHD file path
        :param new_vhd_file_size: Size of the new VHD file.
        :return: Internal VHD size according to new VHD file size.
        """
    try:
        with open(vhd_path, 'rb') as f:
            hs = VHDX_HEADER_SECTION_SIZE
            bes = VHDX_BAT_ENTRY_SIZE
            lss = vhd_info['SectorSize']
            bs = self._get_vhdx_block_size(f)
            ls = self._get_vhdx_log_size(f)
            ms = self._get_vhdx_metadata_size_and_offset(f)[0]
            chunk_ratio = (1 << 23) * lss // bs
            size = new_vhd_file_size
            max_internal_size = bs * chunk_ratio * (size - hs - ls - ms - bes - bes // chunk_ratio) // (bs * chunk_ratio + bes * chunk_ratio + bes)
            return max_internal_size - max_internal_size % bs
    except IOError as ex:
        raise exceptions.VHDException(_('Unable to obtain internal size from VHDX: %(vhd_path)s. Exception: %(ex)s') % {'vhd_path': vhd_path, 'ex': ex})