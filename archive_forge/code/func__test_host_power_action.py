from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def _test_host_power_action(self, action):
    fake_win32 = mock.MagicMock()
    fake_win32.Win32Shutdown = mock.MagicMock()
    self._hostutils._conn_cimv2.Win32_OperatingSystem.return_value = [fake_win32]
    if action == constants.HOST_POWER_ACTION_SHUTDOWN:
        self._hostutils.host_power_action(action)
        fake_win32.Win32Shutdown.assert_called_with(self._hostutils._HOST_FORCED_SHUTDOWN)
    elif action == constants.HOST_POWER_ACTION_REBOOT:
        self._hostutils.host_power_action(action)
        fake_win32.Win32Shutdown.assert_called_with(self._hostutils._HOST_FORCED_REBOOT)
    else:
        self.assertRaises(NotImplementedError, self._hostutils.host_power_action, action)