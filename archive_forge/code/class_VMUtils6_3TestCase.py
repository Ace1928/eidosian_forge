from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
class VMUtils6_3TestCase(test_base.OsWinBaseTestCase):

    def setUp(self):
        super(VMUtils6_3TestCase, self).setUp()
        self._vmutils = vmutils.VMUtils6_3()
        self._vmutils._conn_attr = mock.MagicMock()
        self._vmutils._jobutils = mock.MagicMock()

    @mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
    def test_set_disk_qos_specs(self, mock_get_disk_resource):
        mock_disk = mock_get_disk_resource.return_value
        self._vmutils.set_disk_qos_specs(mock.sentinel.disk_path, max_iops=mock.sentinel.max_iops, min_iops=mock.sentinel.min_iops)
        mock_get_disk_resource.assert_called_once_with(mock.sentinel.disk_path, is_physical=False)
        self.assertEqual(mock.sentinel.max_iops, mock_disk.IOPSLimit)
        self.assertEqual(mock.sentinel.min_iops, mock_disk.IOPSReservation)
        self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(mock_disk)

    @mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
    def test_set_disk_qos_specs_missing_values(self, mock_get_disk_resource):
        self._vmutils.set_disk_qos_specs(mock.sentinel.disk_path)
        self.assertFalse(mock_get_disk_resource.called)