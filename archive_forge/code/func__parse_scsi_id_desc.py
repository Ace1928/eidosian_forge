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
def _parse_scsi_id_desc(self, id_desc_addr, buff_sz):
    """Parse SCSI VPD identification descriptor."""
    id_desc_struct_sz = ctypes.sizeof(IDENTIFICATION_DESCRIPTOR)
    if buff_sz < id_desc_struct_sz:
        reason = _('Identifier descriptor overflow.')
        raise exceptions.SCSIIdDescriptorParsingError(reason=reason)
    id_desc = IDENTIFICATION_DESCRIPTOR.from_address(id_desc_addr)
    id_desc_sz = id_desc_struct_sz + id_desc.IdentifierLength
    identifier_addr = id_desc_addr + id_desc_struct_sz
    if id_desc_sz > buff_sz:
        reason = _('Identifier overflow.')
        raise exceptions.SCSIIdDescriptorParsingError(reason=reason)
    identifier = (ctypes.c_ubyte * id_desc.IdentifierLength).from_address(identifier_addr)
    raw_id = bytearray(identifier)
    if id_desc.CodeSet == SCSI_ID_CODE_SET_ASCII:
        parsed_id = bytes(bytearray(identifier)).decode('ascii').strip('\x00')
    else:
        parsed_id = _utils.byte_array_to_hex_str(raw_id)
    id_dict = {'code_set': id_desc.CodeSet, 'protocol': id_desc.ProtocolIdentifier if id_desc.Piv else None, 'type': id_desc.IdentifierType, 'association': id_desc.Association, 'raw_id': raw_id, 'id': parsed_id, 'raw_id_desc_size': id_desc_sz}
    return id_dict