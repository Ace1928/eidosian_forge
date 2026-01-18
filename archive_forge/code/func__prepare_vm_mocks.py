import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def _prepare_vm_mocks(self, resource_type, resource_sub_type, mock_get_elem_associated_class):
    mock_vm_svc = self._conn.Msvm_VirtualSystemManagementService()[0]
    vm = self._get_vm()
    self._conn.Msvm_PlannedComputerSystem.return_value = [vm]
    mock_vm_svc.DestroySystem.return_value = (mock.sentinel.FAKE_JOB_PATH, self._FAKE_RET_VAL)
    mock_vm_svc.ModifyResourceSettings.return_value = (None, mock.sentinel.FAKE_JOB_PATH, self._FAKE_RET_VAL)
    sasd = mock.MagicMock()
    other_sasd = mock.MagicMock()
    sasd.ResourceType = resource_type
    sasd.ResourceSubType = resource_sub_type
    sasd.HostResource = [mock.sentinel.FAKE_SASD_RESOURCE]
    sasd.path.return_value.RelPath = mock.sentinel.FAKE_DISK_PATH
    mock_get_elem_associated_class.return_value = [sasd, other_sasd]