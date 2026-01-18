from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def _test_get_timeout(self, continue_on_timeout=True):
    self._mock_client_connected.isSet.side_effect = [True, True, False]
    self._mock_queue.get.side_effect = ioutils.Queue.Empty
    queue_item = self._ioqueue.get(timeout=mock.sentinel.timeout, continue_on_timeout=continue_on_timeout)
    expected_calls_number = 2 if continue_on_timeout else 1
    self._mock_queue.get.assert_has_calls([mock.call(self._ioqueue, timeout=mock.sentinel.timeout)] * expected_calls_number)
    self.assertIsNone(queue_item)