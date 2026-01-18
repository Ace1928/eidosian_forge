from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestBackupList(TestBackups):
    columns = database_backups.ListDatabaseBackups.columns
    values = ('bk-1234', '1234', 'bkp_1', 'COMPLETED', None, '2015-05-16T14:23:08', '262db161-d3e4-4218-8bde-5bd879fc3e61')

    def setUp(self):
        super(TestBackupList, self).setUp()
        self.cmd = database_backups.ListDatabaseBackups(self.app, None)
        data = [self.fake_backups.get_backup_bk_1234()]
        self.backup_client.list.return_value = common.Paginated(data)

    def test_backup_list_defaults(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': None, 'all_projects': False, 'project_id': None}
        self.backup_client.list.assert_called_once_with(**params)
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], data)

    @mock.patch('troveclient.utils.get_resource_id')
    def test_backup_list_by_instance_id(self, get_resource_id_mock):
        get_resource_id_mock.return_value = 'fake_uuid'
        parsed_args = self.check_parser(self.cmd, ['--instance-id', 'fake_id'], [])
        self.cmd.take_action(parsed_args)
        params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': 'fake_uuid', 'all_projects': False, 'project_id': None}
        self.backup_client.list.assert_called_once_with(**params)

    @mock.patch('troveclient.utils.get_resource_id')
    def test_backup_list_by_instance_name(self, get_resource_id_mock):
        get_resource_id_mock.return_value = 'fake_uuid'
        parsed_args = self.check_parser(self.cmd, ['--instance', 'fake_name'], [])
        self.cmd.take_action(parsed_args)
        params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': 'fake_uuid', 'all_projects': False, 'project_id': None}
        self.backup_client.list.assert_called_once_with(**params)
        get_resource_id_mock.assert_called_once_with(self.instance_client, 'fake_name')

    def test_backup_list_all_projects(self):
        parsed_args = self.check_parser(self.cmd, ['--all-projects'], [])
        self.cmd.take_action(parsed_args)
        params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': None, 'all_projects': True, 'project_id': None}
        self.backup_client.list.assert_called_once_with(**params)

    def test_backup_list_by_project(self):
        parsed_args = self.check_parser(self.cmd, ['--project-id', 'fake_id'], [])
        self.cmd.take_action(parsed_args)
        params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': None, 'all_projects': False, 'project_id': 'fake_id'}
        self.backup_client.list.assert_called_once_with(**params)