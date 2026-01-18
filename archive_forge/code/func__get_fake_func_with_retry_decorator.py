from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def _get_fake_func_with_retry_decorator(self, side_effect, decorator=_utils.retry_decorator, *args, **kwargs):
    func_side_effect = mock.Mock(side_effect=side_effect)

    @decorator(*args, **kwargs)
    def fake_func(*_args, **_kwargs):
        return func_side_effect(*_args, **_kwargs)
    return (fake_func, func_side_effect)