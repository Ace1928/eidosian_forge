from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils, 'ctypes')
@mock.patch.object(diskutils, 'kernel32', create=True)
@mock.patch('os.path.abspath')
def _test_get_disk_capacity(self, mock_abspath, mock_kernel32, mock_ctypes, raised_exc=None, ignore_errors=False):
    expected_values = ('total_bytes', 'free_bytes')
    mock_params = [mock.Mock(value=value) for value in expected_values]
    mock_ctypes.c_ulonglong.side_effect = mock_params
    mock_ctypes.c_wchar_p = lambda x: (x, 'c_wchar_p')
    self._mock_run.side_effect = raised_exc(func_name='fake_func_name', error_code='fake_error_code', error_message='fake_error_message') if raised_exc else None
    if raised_exc and (not ignore_errors):
        self.assertRaises(raised_exc, self._diskutils.get_disk_capacity, mock.sentinel.disk_path, ignore_errors=ignore_errors)
    else:
        ret_val = self._diskutils.get_disk_capacity(mock.sentinel.disk_path, ignore_errors=ignore_errors)
        expected_ret_val = (0, 0) if raised_exc else expected_values
        self.assertEqual(expected_ret_val, ret_val)
    mock_abspath.assert_called_once_with(mock.sentinel.disk_path)
    mock_ctypes.pointer.assert_has_calls([mock.call(param) for param in mock_params])
    self._mock_run.assert_called_once_with(mock_kernel32.GetDiskFreeSpaceExW, mock_ctypes.c_wchar_p(mock_abspath.return_value), None, mock_ctypes.pointer.return_value, mock_ctypes.pointer.return_value, kernel32_lib_func=True)