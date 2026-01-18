from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestListMigrationV280(TestListMigration):
    """Test fetch all migrations by user-id and/or project-id."""
    MIGRATION_COLUMNS = ['Id', 'UUID', 'Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Old Flavor', 'New Flavor', 'Type', 'Created At', 'Updated At']
    MIGRATION_FIELDS = ['id', 'uuid', 'source_node', 'dest_node', 'source_compute', 'dest_compute', 'dest_host', 'status', 'server_id', 'old_flavor_id', 'new_flavor_id', 'migration_type', 'created_at', 'updated_at']
    project = identity_fakes.FakeProject.create_one_project()
    user = identity_fakes.FakeUser.create_one_user()

    def setUp(self):
        super().setUp()
        self.projects_mock = self.app.client_manager.identity.projects
        self.projects_mock.reset_mock()
        self.users_mock = self.app.client_manager.identity.users
        self.users_mock.reset_mock()
        self.projects_mock.get.return_value = self.project
        self.users_mock.get.return_value = self.user
        self._set_mock_microversion('2.80')

    def test_server_migration_list_with_project(self):
        arglist = ['--status', 'migrating', '--limit', '1', '--marker', 'test_kp', '--changes-since', '2019-08-07T08:03:25Z', '--changes-before', '2019-08-09T08:03:25Z', '--project', self.project.id]
        verifylist = [('status', 'migrating'), ('limit', 1), ('marker', 'test_kp'), ('changes_since', '2019-08-07T08:03:25Z'), ('changes_before', '2019-08-09T08:03:25Z'), ('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'status': 'migrating', 'limit': 1, 'paginated': False, 'marker': 'test_kp', 'project_id': self.project.id, 'changes_since': '2019-08-07T08:03:25Z', 'changes_before': '2019-08-09T08:03:25Z'}
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.MIGRATION_COLUMNS.insert(len(self.MIGRATION_COLUMNS) - 2, 'Project')
        self.MIGRATION_FIELDS.insert(len(self.MIGRATION_FIELDS) - 2, 'project_id')
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))
        self.MIGRATION_COLUMNS.remove('Project')
        self.MIGRATION_FIELDS.remove('project_id')

    def test_get_migrations_with_project_pre_v280(self):
        self._set_mock_microversion('2.79')
        arglist = ['--status', 'migrating', '--changes-before', '2019-08-09T08:03:25Z', '--project', '0c2accde-644a-45fa-8c10-e76debc7fbc3']
        verifylist = [('status', 'migrating'), ('changes_before', '2019-08-09T08:03:25Z'), ('project', '0c2accde-644a-45fa-8c10-e76debc7fbc3')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.80 or greater is required', str(ex))

    def test_server_migration_list_with_user(self):
        arglist = ['--status', 'migrating', '--limit', '1', '--marker', 'test_kp', '--changes-since', '2019-08-07T08:03:25Z', '--changes-before', '2019-08-09T08:03:25Z', '--user', self.user.id]
        verifylist = [('status', 'migrating'), ('limit', 1), ('marker', 'test_kp'), ('changes_since', '2019-08-07T08:03:25Z'), ('changes_before', '2019-08-09T08:03:25Z'), ('user', self.user.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'status': 'migrating', 'limit': 1, 'paginated': False, 'marker': 'test_kp', 'user_id': self.user.id, 'changes_since': '2019-08-07T08:03:25Z', 'changes_before': '2019-08-09T08:03:25Z'}
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.MIGRATION_COLUMNS.insert(len(self.MIGRATION_COLUMNS) - 2, 'User')
        self.MIGRATION_FIELDS.insert(len(self.MIGRATION_FIELDS) - 2, 'user_id')
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))
        self.MIGRATION_COLUMNS.remove('User')
        self.MIGRATION_FIELDS.remove('user_id')

    def test_get_migrations_with_user_pre_v280(self):
        self._set_mock_microversion('2.79')
        arglist = ['--status', 'migrating', '--changes-before', '2019-08-09T08:03:25Z', '--user', self.user.id]
        verifylist = [('status', 'migrating'), ('changes_before', '2019-08-09T08:03:25Z'), ('user', self.user.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.80 or greater is required', str(ex))

    def test_server_migration_list_with_project_and_user(self):
        arglist = ['--status', 'migrating', '--limit', '1', '--changes-since', '2019-08-07T08:03:25Z', '--changes-before', '2019-08-09T08:03:25Z', '--project', self.project.id, '--user', self.user.id]
        verifylist = [('status', 'migrating'), ('limit', 1), ('changes_since', '2019-08-07T08:03:25Z'), ('changes_before', '2019-08-09T08:03:25Z'), ('project', self.project.id), ('user', self.user.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'status': 'migrating', 'limit': 1, 'paginated': False, 'project_id': self.project.id, 'user_id': self.user.id, 'changes_since': '2019-08-07T08:03:25Z', 'changes_before': '2019-08-09T08:03:25Z'}
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.MIGRATION_COLUMNS.insert(len(self.MIGRATION_COLUMNS) - 2, 'Project')
        self.MIGRATION_FIELDS.insert(len(self.MIGRATION_FIELDS) - 2, 'project_id')
        self.MIGRATION_COLUMNS.insert(len(self.MIGRATION_COLUMNS) - 2, 'User')
        self.MIGRATION_FIELDS.insert(len(self.MIGRATION_FIELDS) - 2, 'user_id')
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))
        self.MIGRATION_COLUMNS.remove('Project')
        self.MIGRATION_FIELDS.remove('project_id')
        self.MIGRATION_COLUMNS.remove('User')
        self.MIGRATION_FIELDS.remove('user_id')

    def test_get_migrations_with_project_and_user_pre_v280(self):
        self._set_mock_microversion('2.79')
        arglist = ['--status', 'migrating', '--changes-before', '2019-08-09T08:03:25Z', '--project', self.project.id, '--user', self.user.id]
        verifylist = [('status', 'migrating'), ('changes_before', '2019-08-09T08:03:25Z'), ('project', self.project.id), ('user', self.user.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.80 or greater is required', str(ex))