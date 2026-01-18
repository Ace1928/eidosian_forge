from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_new_resource_setting_data')
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
def _test_attach_volume_to_controller(self, mock_get_wmi_obj, mock_get_new_rsd, disk_serial=None):
    mock_vm = self._lookup_vm()
    mock_diskdrive = mock.MagicMock()
    jobutils = self._vmutils._jobutils
    jobutils.add_virt_resource.return_value = [mock_diskdrive]
    mock_get_wmi_obj.return_value = mock_diskdrive
    self._vmutils.attach_volume_to_controller(self._FAKE_VM_NAME, self._FAKE_CTRL_PATH, self._FAKE_CTRL_ADDR, self._FAKE_MOUNTED_DISK_PATH, serial=disk_serial)
    self._vmutils._jobutils.add_virt_resource.assert_called_once_with(mock_get_new_rsd.return_value, mock_vm)
    if disk_serial:
        jobutils.modify_virt_resource.assert_called_once_with(mock_diskdrive)
        self.assertEqual(disk_serial, mock_diskdrive.ElementName)