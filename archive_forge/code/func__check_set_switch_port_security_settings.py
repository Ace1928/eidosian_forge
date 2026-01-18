from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_security_setting_data_from_port_alloc')
@mock.patch.object(networkutils.NetworkUtils, '_create_default_setting_data')
def _check_set_switch_port_security_settings(self, mock_create_default_sd, mock_get_security_sd, missing_sec=False):
    mock_port_alloc = self._mock_get_switch_port_alloc()
    mock_sec_settings = mock.MagicMock()
    mock_get_security_sd.return_value = None if missing_sec else mock_sec_settings
    mock_create_default_sd.return_value = mock_sec_settings
    if missing_sec:
        self.assertRaises(exceptions.HyperVException, self.netutils._set_switch_port_security_settings, mock.sentinel.switch_port_name, VirtualSubnetId=mock.sentinel.vsid)
        mock_create_default_sd.assert_called_once_with(self.netutils._PORT_SECURITY_SET_DATA)
    else:
        self.netutils._set_switch_port_security_settings(mock.sentinel.switch_port_name, VirtualSubnetId=mock.sentinel.vsid)
    self.assertEqual(mock.sentinel.vsid, mock_sec_settings.VirtualSubnetId)
    if missing_sec:
        mock_add_feature = self.netutils._jobutils.add_virt_feature
        mock_add_feature.assert_called_once_with(mock_sec_settings, mock_port_alloc)
    else:
        mock_modify_feature = self.netutils._jobutils.modify_virt_feature
        mock_modify_feature.assert_called_once_with(mock_sec_settings)