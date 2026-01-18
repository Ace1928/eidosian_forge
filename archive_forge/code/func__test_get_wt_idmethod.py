from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def _test_get_wt_idmethod(self, idmeth_found=True):
    mock_wt_idmeth = mock.Mock()
    mock_wt_idmeth_cls = self._tgutils._conn_wmi.WT_IDMethod
    mock_wt_idmeth_cls.return_value = [mock_wt_idmeth] if idmeth_found else []
    wt_idmeth = self._tgutils._get_wt_idmethod(mock.sentinel.initiator, mock.sentinel.target_name)
    expected_wt_idmeth = mock_wt_idmeth if idmeth_found else None
    self.assertEqual(expected_wt_idmeth, wt_idmeth)
    mock_wt_idmeth_cls.assert_called_once_with(HostName=mock.sentinel.target_name, Value=mock.sentinel.initiator)