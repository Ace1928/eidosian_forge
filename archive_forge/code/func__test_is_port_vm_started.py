from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def _test_is_port_vm_started(self, vm_state, expected_result):
    mock_svc = self.netutils._conn.Msvm_VirtualSystemManagementService()[0]
    mock_port = mock.MagicMock()
    mock_vmsettings = mock.MagicMock()
    mock_summary = mock.MagicMock()
    mock_summary.EnabledState = vm_state
    mock_vmsettings.path_.return_value = self._FAKE_RES_PATH
    self.netutils._conn.Msvm_VirtualSystemSettingData.return_value = [mock_vmsettings]
    mock_svc.GetSummaryInformation.return_value = (self._FAKE_RET_VAL, [mock_summary])
    result = self.netutils._is_port_vm_started(mock_port)
    self.assertEqual(expected_result, result)
    mock_svc.GetSummaryInformation.assert_called_once_with([self.netutils._VM_SUMMARY_ENABLED_STATE], [self._FAKE_RES_PATH])