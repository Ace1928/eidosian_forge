from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
@mock.patch.object(smbutils.SMBUtils, 'unmount_smb_share')
@mock.patch('os.path.exists')
def _test_check_smb_mapping(self, mock_exists, mock_unmount_smb_share, existing_mappings=True, share_available=False):
    mock_exists.return_value = share_available
    fake_mappings = [mock.sentinel.smb_mapping] if existing_mappings else []
    self._smb_conn.Msft_SmbMapping.return_value = fake_mappings
    ret_val = self._smbutils.check_smb_mapping(mock.sentinel.share_path, remove_unavailable_mapping=True)
    self.assertEqual(existing_mappings and share_available, ret_val)
    if existing_mappings and (not share_available):
        mock_unmount_smb_share.assert_called_once_with(mock.sentinel.share_path, force=True)