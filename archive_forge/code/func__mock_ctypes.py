import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def _mock_ctypes(self):
    self._ctypes = mock.Mock()
    self._ctypes.byref = lambda x: (x, 'byref')
    mock.patch.object(iscsi_utils, 'ctypes', self._ctypes).start()