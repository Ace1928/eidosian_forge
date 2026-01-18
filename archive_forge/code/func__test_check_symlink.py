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
@mock.patch('os.path.isdir')
@mock.patch('os.path.islink')
def _test_check_symlink(self, mock_is_symlink, mock_is_dir, is_symlink=True, is_dir=True):
    fake_path = 'c:\\\\fake_path'
    if is_symlink:
        f_attr = 1024
    else:
        f_attr = 128
    mock_is_dir.return_value = is_dir
    mock_is_symlink.return_value = is_symlink
    self._mock_run.return_value = f_attr
    ret_value = self._pathutils.is_symlink(fake_path)
    mock_is_symlink.assert_called_once_with(fake_path)
    self.assertEqual(is_symlink, ret_value)