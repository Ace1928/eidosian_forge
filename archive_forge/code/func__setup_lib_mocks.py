import ctypes
import os
from unittest import mock
import uuid
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.virtdisk import vhdutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def _setup_lib_mocks(self):
    self._vdisk_struct = mock.Mock()
    self._ctypes = mock.Mock()
    self._ctypes.byref = lambda x: (x, 'byref')
    self._ctypes.c_wchar_p = lambda x: (x, 'c_wchar_p')
    self._ctypes.c_ulong = lambda x: (x, 'c_ulong')
    self._ctypes_patcher = mock.patch.object(vhdutils, 'ctypes', self._ctypes)
    self._ctypes_patcher.start()
    mock.patch.multiple(vhdutils, kernel32=mock.DEFAULT, wintypes=mock.DEFAULT, virtdisk=mock.DEFAULT, vdisk_struct=self._vdisk_struct, create=True).start()