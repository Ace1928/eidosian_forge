from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def _test_connect_vnic_to_vswitch(self, found):
    self.netutils._get_vnic_settings = mock.MagicMock()
    if not found:
        mock_vm = mock.MagicMock()
        self.netutils._get_vm_from_res_setting_data = mock.MagicMock(return_value=mock_vm)
        self.netutils._add_virt_resource = mock.MagicMock()
    else:
        self.netutils._modify_virt_resource = mock.MagicMock()
    self.netutils._get_vswitch = mock.MagicMock()
    mock_port = self._mock_get_switch_port_alloc(found=found)
    mock_port.HostResource = []
    self.netutils.connect_vnic_to_vswitch(self._FAKE_VSWITCH_NAME, self._FAKE_PORT_NAME)
    if not found:
        mock_add_resource = self.netutils._jobutils.add_virt_resource
        mock_add_resource.assert_called_once_with(mock_port, mock_vm)
    else:
        mock_modify_resource = self.netutils._jobutils.modify_virt_resource
        mock_modify_resource.assert_called_once_with(mock_port)