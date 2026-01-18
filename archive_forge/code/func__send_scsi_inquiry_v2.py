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
def _send_scsi_inquiry_v2(self, hba_handle, port_wwn_struct, remote_port_wwn_struct, fcp_lun, cdb_byte1, cdb_byte2):
    port_wwn = _utils.byte_array_to_hex_str(port_wwn_struct.wwn)
    remote_port_wwn = _utils.byte_array_to_hex_str(remote_port_wwn_struct.wwn)
    LOG.debug('Sending SCSI INQUIRY to WWPN %(remote_port_wwn)s, FCP LUN %(fcp_lun)s from WWPN %(port_wwn)s. CDB byte 1 %(cdb_byte1)s, CDB byte 2: %(cdb_byte2)s.', dict(port_wwn=port_wwn, remote_port_wwn=remote_port_wwn, fcp_lun=fcp_lun, cdb_byte1=hex(cdb_byte1), cdb_byte2=hex(cdb_byte2)))
    resp_buffer_sz = ctypes.c_uint32(SCSI_INQ_BUFF_SZ)
    resp_buffer = (ctypes.c_ubyte * resp_buffer_sz.value)()
    sense_buffer_sz = ctypes.c_uint32(SENSE_BUFF_SZ)
    sense_buffer = (ctypes.c_ubyte * sense_buffer_sz.value)()
    scsi_status = ctypes.c_ubyte()
    try:
        self._run_and_check_output(hbaapi.HBA_ScsiInquiryV2, hba_handle, port_wwn_struct, remote_port_wwn_struct, ctypes.c_uint64(fcp_lun), ctypes.c_uint8(cdb_byte1), ctypes.c_uint8(cdb_byte2), ctypes.byref(resp_buffer), ctypes.byref(resp_buffer_sz), ctypes.byref(scsi_status), ctypes.byref(sense_buffer), ctypes.byref(sense_buffer_sz))
    finally:
        sense_data = _utils.byte_array_to_hex_str(sense_buffer[:sense_buffer_sz.value])
        LOG.debug('SCSI inquiry returned sense data: %(sense_data)s. SCSI status: %(scsi_status)s.', dict(sense_data=sense_data, scsi_status=scsi_status.value))
    return resp_buffer