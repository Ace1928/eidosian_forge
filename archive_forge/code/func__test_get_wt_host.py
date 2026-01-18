from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def _test_get_wt_host(self, host_found=True, fail_if_not_found=False):
    mock_wt_host = mock.Mock()
    mock_wt_host_cls = self._tgutils._conn_wmi.WT_Host
    mock_wt_host_cls.return_value = [mock_wt_host] if host_found else []
    if not host_found and fail_if_not_found:
        self.assertRaises(exceptions.ISCSITargetException, self._tgutils._get_wt_host, mock.sentinel.target_name, fail_if_not_found=fail_if_not_found)
    else:
        wt_host = self._tgutils._get_wt_host(mock.sentinel.target_name, fail_if_not_found=fail_if_not_found)
        expected_wt_host = mock_wt_host if host_found else None
        self.assertEqual(expected_wt_host, wt_host)
    mock_wt_host_cls.assert_called_once_with(HostName=mock.sentinel.target_name)