from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def _create_mock_binding(self):
    binding = mock.MagicMock()
    binding.BindName = self.utils._WNV_BIND_NAME
    binding.Name = mock.sentinel.fake_network
    net_binds = self.utils._scimv2.MSFT_NetAdapterBindingSettingData
    net_binds.return_value = [binding]
    return binding