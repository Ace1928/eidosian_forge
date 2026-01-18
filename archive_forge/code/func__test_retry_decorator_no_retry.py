from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch('time.sleep')
def _test_retry_decorator_no_retry(self, mock_sleep, expected_exceptions=(), expected_error_codes=()):
    err_code = 1
    raised_exc = exceptions.Win32Exception(message='fake_exc', error_code=err_code)
    fake_func, fake_func_side_effect = self._get_fake_func_with_retry_decorator(error_codes=expected_error_codes, exceptions=expected_exceptions, side_effect=raised_exc)
    self.assertRaises(exceptions.Win32Exception, fake_func, mock.sentinel.arg, fake_kwarg=mock.sentinel.kwarg)
    self.assertFalse(mock_sleep.called)
    fake_func_side_effect.assert_called_once_with(mock.sentinel.arg, fake_kwarg=mock.sentinel.kwarg)