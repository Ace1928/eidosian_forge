from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch.object(_utils, 'time')
def _test_retry_decorator_exceeded(self, mock_time, expected_try_count, mock_time_side_eff=None, timeout=None, max_retry_count=None):
    raised_exc = exceptions.Win32Exception(message='fake_exc')
    mock_time.time.side_effect = mock_time_side_eff
    fake_func, fake_func_side_effect = self._get_fake_func_with_retry_decorator(exceptions=exceptions.Win32Exception, timeout=timeout, side_effect=raised_exc)
    self.assertRaises(exceptions.Win32Exception, fake_func)
    fake_func_side_effect.assert_has_calls([mock.call()] * expected_try_count)