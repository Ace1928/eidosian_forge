from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def _check_set_vm_vcpus(self, mock_get_element_associated_class, vcpus_per_numa_node=None):
    procsetting = mock.MagicMock()
    mock_vmsettings = mock.MagicMock()
    mock_get_element_associated_class.return_value = [procsetting]
    self._vmutils._set_vm_vcpus(mock_vmsettings, self._FAKE_VCPUS_NUM, vcpus_per_numa_node, limit_cpu_features=False)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(procsetting)
    if vcpus_per_numa_node:
        self.assertEqual(vcpus_per_numa_node, procsetting.MaxProcessorsPerNumaNode)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._conn, self._vmutils._PROCESSOR_SETTING_DATA_CLASS, element_instance_id=mock_vmsettings.InstanceID)