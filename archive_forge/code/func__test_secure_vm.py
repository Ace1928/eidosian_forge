from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
@mock.patch.object(vmutils10.VMUtils10, 'get_vm_id')
def _test_secure_vm(self, mock_get_vm_id, mock_get_element_associated_class, is_encrypted_vm=True):
    inst_id = mock_get_vm_id.return_value
    security_profile = mock.MagicMock()
    mock_get_element_associated_class.return_value = [security_profile]
    security_profile.EncryptStateAndVmMigrationTraffic = is_encrypted_vm
    response = self._vmutils.is_secure_vm(mock.sentinel.instance_name)
    self.assertEqual(is_encrypted_vm, response)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._conn, self._vmutils._SECURITY_SETTING_DATA, element_uuid=inst_id)