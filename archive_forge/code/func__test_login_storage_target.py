import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_login_opts')
@mock.patch.object(iscsi_struct, 'ISCSI_TARGET_PORTAL')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_new_session_required')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, 'get_targets')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_login_iscsi_target')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, 'ensure_lun_available')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_add_static_target')
def _test_login_storage_target(self, mock_add_static_target, mock_ensure_lun_available, mock_login_iscsi_target, mock_get_targets, mock_session_required, mock_cls_ISCSI_TARGET_PORTAL, mock_get_login_opts, mpio_enabled=False, login_required=True):
    fake_portal_addr = '127.0.0.1'
    fake_portal_port = 3260
    fake_target_portal = '%s:%s' % (fake_portal_addr, fake_portal_port)
    fake_portal = mock_cls_ISCSI_TARGET_PORTAL.return_value
    fake_login_opts = mock_get_login_opts.return_value
    mock_get_targets.return_value = []
    mock_login_iscsi_target.return_value = (mock.sentinel.session_id, mock.sentinel.conn_id)
    mock_session_required.return_value = login_required
    self._initiator.login_storage_target(mock.sentinel.target_lun, mock.sentinel.target_iqn, fake_target_portal, auth_username=mock.sentinel.auth_username, auth_password=mock.sentinel.auth_password, auth_type=mock.sentinel.auth_type, mpio_enabled=mpio_enabled, rescan_attempts=mock.sentinel.rescan_attempts)
    mock_get_targets.assert_called_once_with()
    mock_add_static_target.assert_called_once_with(mock.sentinel.target_iqn)
    if login_required:
        expected_login_flags = w_const.ISCSI_LOGIN_FLAG_MULTIPATH_ENABLED if mpio_enabled else 0
        mock_get_login_opts.assert_called_once_with(mock.sentinel.auth_username, mock.sentinel.auth_password, mock.sentinel.auth_type, expected_login_flags)
        mock_cls_ISCSI_TARGET_PORTAL.assert_called_once_with(Address=fake_portal_addr, Socket=fake_portal_port)
        mock_login_iscsi_target.assert_has_calls([mock.call(mock.sentinel.target_iqn, fake_portal, fake_login_opts, is_persistent=True), mock.call(mock.sentinel.target_iqn, fake_portal, fake_login_opts, is_persistent=False)])
    else:
        self.assertFalse(mock_login_iscsi_target.called)
    mock_ensure_lun_available.assert_called_once_with(mock.sentinel.target_iqn, mock.sentinel.target_lun, mock.sentinel.rescan_attempts)