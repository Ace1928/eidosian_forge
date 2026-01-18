from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(ioutils.IOQueue, 'get')
def _test_get_burst(self, mock_get, exceeded_max_size=False):
    fake_data = 'fake_data'
    mock_get.side_effect = [fake_data, fake_data, None]
    if exceeded_max_size:
        max_size = 0
    else:
        max_size = constants.SERIAL_CONSOLE_BUFFER_SIZE
    ret_val = self._ioqueue.get_burst(timeout=mock.sentinel.timeout, burst_timeout=mock.sentinel.burst_timeout, max_size=max_size)
    expected_calls = [mock.call(timeout=mock.sentinel.timeout)]
    expected_ret_val = fake_data
    if not exceeded_max_size:
        expected_calls.append(mock.call(timeout=mock.sentinel.burst_timeout, continue_on_timeout=False))
        expected_ret_val += fake_data
    mock_get.assert_has_calls(expected_calls)
    self.assertEqual(expected_ret_val, ret_val)