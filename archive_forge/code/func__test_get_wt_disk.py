from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def _test_get_wt_disk(self, disk_found=True, fail_if_not_found=False):
    mock_wt_disk = mock.Mock()
    mock_wt_disk_cls = self._tgutils._conn_wmi.WT_Disk
    mock_wt_disk_cls.return_value = [mock_wt_disk] if disk_found else []
    if not disk_found and fail_if_not_found:
        self.assertRaises(exceptions.ISCSITargetException, self._tgutils._get_wt_disk, mock.sentinel.disk_description, fail_if_not_found=fail_if_not_found)
    else:
        wt_disk = self._tgutils._get_wt_disk(mock.sentinel.disk_description, fail_if_not_found=fail_if_not_found)
        expected_wt_disk = mock_wt_disk if disk_found else None
        self.assertEqual(expected_wt_disk, wt_disk)
    mock_wt_disk_cls.assert_called_once_with(Description=mock.sentinel.disk_description)