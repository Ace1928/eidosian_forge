import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def _wwn_struct_from_hex_str(self, wwn_hex_str):
    try:
        wwn_struct = fc_struct.HBA_WWN()
        wwn_struct.wwn[:] = _utils.hex_str_to_byte_array(wwn_hex_str)
    except ValueError:
        err_msg = _('Invalid WWN hex string received: %s') % wwn_hex_str
        raise exceptions.FCException(err_msg)
    return wwn_struct