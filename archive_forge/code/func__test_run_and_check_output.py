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
def _test_run_and_check_output(self, raised_exc=None):
    self._mock_run.side_effect = raised_exc(func_name='fake_func_name', error_code='fake_error_code', error_message='fake_error_message') if raised_exc else None
    if raised_exc:
        self.assertRaises(raised_exc, self._vhdutils._run_and_check_output, mock.sentinel.func, mock.sentinel.arg, cleanup_handle=mock.sentinel.handle)
    else:
        ret_val = self._vhdutils._run_and_check_output(mock.sentinel.func, mock.sentinel.arg, cleanup_handle=mock.sentinel.handle)
        self.assertEqual(self._mock_run.return_value, ret_val)
    self._mock_run.assert_called_once_with(mock.sentinel.func, mock.sentinel.arg, **self._run_args)
    self._mock_close.assert_called_once_with(mock.sentinel.handle)