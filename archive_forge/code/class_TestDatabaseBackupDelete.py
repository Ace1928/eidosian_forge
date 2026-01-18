from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestDatabaseBackupDelete(TestBackups):

    def setUp(self):
        super(TestDatabaseBackupDelete, self).setUp()
        self.cmd = database_backups.DeleteDatabaseBackup(self.app, None)

    @mock.patch('troveclient.utils.get_resource_id_by_name')
    def test_backup_delete(self, mock_getid):
        args = ['backup1']
        mock_getid.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.backup_client.delete.assert_called_with('backup1')

    @mock.patch('troveclient.utils.get_resource_id_by_name')
    def test_backup_delete_with_exception(self, mock_getid):
        args = ['fakebackup']
        parsed_args = self.check_parser(self.cmd, args, [])
        mock_getid.side_effect = exceptions.CommandError
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch('troveclient.utils.get_resource_id_by_name')
    def test_backup_bulk_delete(self, mock_getid):
        backup_1 = uuidutils.generate_uuid()
        backup_2 = uuidutils.generate_uuid()
        mock_getid.return_value = backup_1
        args = ['fake_backup', backup_2]
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        mock_getid.assert_called_once_with(self.backup_client, 'fake_backup')
        calls = [mock.call(backup_1), mock.call(backup_2)]
        self.backup_client.delete.assert_has_calls(calls)