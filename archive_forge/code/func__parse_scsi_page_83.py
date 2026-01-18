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
def _parse_scsi_page_83(self, buff, select_supported_identifiers=False):
    """Parse SCSI Device Identification VPD (page 0x83 data).

        :param buff: a byte array containing the SCSI page 0x83 data.
        :param select_supported_identifiers: select identifiers supported
            by Windows, in the order of precedence.
        :returns: a list of identifiers represented as dicts, containing
                  SCSI Unique IDs.
        """
    identifiers = []
    buff_sz = len(buff)
    buff = (ctypes.c_ubyte * buff_sz)(*bytearray(buff))
    vpd_pg_struct_sz = ctypes.sizeof(DEVICE_ID_VPD_PAGE)
    if buff_sz < vpd_pg_struct_sz:
        reason = _('Invalid VPD page data.')
        raise exceptions.SCSIPageParsingError(page='0x83', reason=reason)
    vpd_page = ctypes.cast(buff, PDEVICE_ID_VPD_PAGE).contents
    vpd_page_addr = ctypes.addressof(vpd_page)
    total_page_sz = vpd_page.PageLength + vpd_pg_struct_sz
    if vpd_page.PageCode != 131:
        reason = _('Unexpected page code: %s') % vpd_page.PageCode
        raise exceptions.SCSIPageParsingError(page='0x83', reason=reason)
    if total_page_sz > buff_sz:
        reason = _('VPD page overflow.')
        raise exceptions.SCSIPageParsingError(page='0x83', reason=reason)
    if not vpd_page.PageLength:
        LOG.info('Page 0x83 data does not contain any identification descriptors.')
        return identifiers
    id_desc_offset = vpd_pg_struct_sz
    while id_desc_offset < total_page_sz:
        id_desc_addr = vpd_page_addr + id_desc_offset
        id_desc_buff_sz = buff_sz - id_desc_offset
        identifier = self._parse_scsi_id_desc(id_desc_addr, id_desc_buff_sz)
        identifiers.append(identifier)
        id_desc_offset += identifier['raw_id_desc_size']
    if select_supported_identifiers:
        identifiers = self._select_supported_scsi_identifiers(identifiers)
    return identifiers