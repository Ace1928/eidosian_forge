import ctypes
import os
import shutil
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import pathutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def _get_file_id_info(self, volume_id, file_id, as_dict=False):
    identifier = (wintypes.BYTE * 16)()
    assert file_id < 1 << 128
    idx = 0
    while file_id:
        identifier[idx] = file_id & 65535
        file_id >>= 8
        idx += 1
    file_id_info = kernel32_def.FILE_ID_INFO(VolumeSerialNumber=volume_id, FileId=kernel32_def.FILE_ID_128(Identifier=identifier))
    if as_dict:
        return dict(volume_serial_number=file_id_info.VolumeSerialNumber, file_id=bytearray(file_id_info.FileId.Identifier))
    return file_id_info