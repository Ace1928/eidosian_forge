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
def _get_mock_file_handle(self, *args):
    mock_file_handle = mock.Mock()
    mock_file_handle.read.side_effect = args
    return mock_file_handle