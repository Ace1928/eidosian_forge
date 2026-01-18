from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestBackupCreate(TestBackups):
    values = ('2015-05-16T14:22:28', 'mysql', '5.6', 'v-56', None, 'bk-1234', '1234', 'http://backup_srvr/database_backups/bk-1234.xbstream.gz.enc', 'bkp_1', None, '262db161-d3e4-4218-8bde-5bd879fc3e61', 0.11, 'COMPLETED', '2015-05-16T14:23:08')

    def setUp(self):
        super(TestBackupCreate, self).setUp()
        self.cmd = database_backups.CreateDatabaseBackup(self.app, None)
        self.data = self.fake_backups.get_backup_bk_1234()
        self.backup_client.create.return_value = self.data
        self.columns = ('created', 'datastore', 'datastore_version', 'datastore_version_id', 'description', 'id', 'instance_id', 'locationRef', 'name', 'parent_id', 'project_id', 'size', 'status', 'updated')

    def test_backup_create_return_value(self):
        args = ['bk-1234', '--instance', self.random_uuid()]
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)

    @mock.patch('troveclient.utils.get_resource_id_by_name')
    def test_backup_create(self, mock_find):
        args = ['bk-1234-1', '--instance', '1234']
        mock_find.return_value = 'fake-instance-id'
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.backup_client.create.assert_called_with('bk-1234-1', 'fake-instance-id', description=None, parent_id=None, incremental=False, swift_container=None)

    @mock.patch('troveclient.utils.get_resource_id_by_name')
    def test_incremental_backup_create(self, mock_find):
        args = ['bk-1234-2', '--instance', '1234', '--description', 'backup 1234', '--parent', '1234-1', '--incremental']
        mock_find.return_value = 'fake-instance-id'
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.backup_client.create.assert_called_with('bk-1234-2', 'fake-instance-id', description='backup 1234', parent_id='1234-1', incremental=True, swift_container=None)

    def test_create_from_data_location(self):
        name = self.random_name('backup')
        ds_version = self.random_uuid()
        args = [name, '--restore-from', 'fake-remote-location', '--restore-datastore-version', ds_version, '--restore-size', '3']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.backup_client.create.assert_called_with(name, None, restore_from='fake-remote-location', restore_ds_version=ds_version, restore_size=3)

    def test_required_params_missing(self):
        args = [self.random_name('backup')]
        parsed_args = self.check_parser(self.cmd, args, [])
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)