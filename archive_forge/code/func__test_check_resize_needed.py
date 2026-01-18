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
@mock.patch.object(vhdutils.VHDUtils, 'get_vhd_size')
def _test_check_resize_needed(self, mock_get_vhd_size, current_size=1, new_size=2):
    mock_get_vhd_size.return_value = dict(VirtualSize=current_size)
    if current_size > new_size:
        self.assertRaises(exceptions.VHDException, self._vhdutils._check_resize_needed, mock.sentinel.vhd_path, new_size)
    else:
        resize_needed = self._vhdutils._check_resize_needed(mock.sentinel.vhd_path, new_size)
        self.assertEqual(current_size < new_size, resize_needed)