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
def _get_scsi_device_id_vpd(self, hba_handle, port_wwn_struct, remote_port_wwn_struct, fcp_lun):
    cdb_byte1 = 1
    cdb_byte2 = 131
    return self._send_scsi_inquiry_v2(hba_handle, port_wwn_struct, remote_port_wwn_struct, fcp_lun, cdb_byte1, cdb_byte2)