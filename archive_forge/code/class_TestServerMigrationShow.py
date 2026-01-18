from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestServerMigrationShow(TestServerMigration):

    def setUp(self):
        super().setUp()
        self.server = compute_fakes.create_one_sdk_server()
        self.compute_sdk_client.find_server.return_value = self.server
        self.server_migration = compute_fakes.create_one_server_migration()
        self.compute_sdk_client.get_server_migration.return_value = self.server_migration
        self.compute_sdk_client.server_migrations.return_value = iter([self.server_migration])
        self.columns = ('ID', 'Server UUID', 'Status', 'Source Compute', 'Source Node', 'Dest Compute', 'Dest Host', 'Dest Node', 'Memory Total Bytes', 'Memory Processed Bytes', 'Memory Remaining Bytes', 'Disk Total Bytes', 'Disk Processed Bytes', 'Disk Remaining Bytes', 'Created At', 'Updated At')
        self.data = (self.server_migration.id, self.server_migration.server_id, self.server_migration.status, self.server_migration.source_compute, self.server_migration.source_node, self.server_migration.dest_compute, self.server_migration.dest_host, self.server_migration.dest_node, self.server_migration.memory_total_bytes, self.server_migration.memory_processed_bytes, self.server_migration.memory_remaining_bytes, self.server_migration.disk_total_bytes, self.server_migration.disk_processed_bytes, self.server_migration.disk_remaining_bytes, self.server_migration.created_at, self.server_migration.updated_at)
        self.cmd = server_migration.ShowMigration(self.app, None)

    def _test_server_migration_show(self):
        arglist = [self.server.id, '2']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)
        self.compute_sdk_client.find_server.assert_called_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.get_server_migration.assert_called_with(self.server.id, '2', ignore_missing=False)

    def test_server_migration_show(self):
        self._set_mock_microversion('2.24')
        self._test_server_migration_show()

    def test_server_migration_show_v259(self):
        self._set_mock_microversion('2.59')
        self.columns += ('UUID',)
        self.data += (self.server_migration.uuid,)
        self._test_server_migration_show()

    def test_server_migration_show_v280(self):
        self._set_mock_microversion('2.80')
        self.columns += ('UUID', 'User ID', 'Project ID')
        self.data += (self.server_migration.uuid, self.server_migration.user_id, self.server_migration.project_id)
        self._test_server_migration_show()

    def test_server_migration_show_pre_v224(self):
        self._set_mock_microversion('2.23')
        arglist = [self.server.id, '2']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.24 or greater is required', str(ex))

    def test_server_migration_show_by_uuid(self):
        self._set_mock_microversion('2.59')
        self.compute_sdk_client.server_migrations.return_value = iter([self.server_migration])
        self.columns += ('UUID',)
        self.data += (self.server_migration.uuid,)
        arglist = [self.server.id, self.server_migration.uuid]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)
        self.compute_sdk_client.find_server.assert_called_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.server_migrations.assert_called_with(self.server.id)
        self.compute_sdk_client.get_server_migration.assert_not_called()

    def test_server_migration_show_by_uuid_no_matches(self):
        self._set_mock_microversion('2.59')
        self.compute_sdk_client.server_migrations.return_value = iter([])
        arglist = [self.server.id, '69f95745-bfe3-4302-90f7-5b0022cba1ce']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('In-progress live migration 69f95745-bfe3-4302-90f7-5b0022cba1ce', str(ex))

    def test_server_migration_show_by_uuid_pre_v259(self):
        self._set_mock_microversion('2.58')
        arglist = [self.server.id, '69f95745-bfe3-4302-90f7-5b0022cba1ce']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.59 or greater is required', str(ex))

    def test_server_migration_show_invalid_id(self):
        self._set_mock_microversion('2.24')
        arglist = [self.server.id, 'foo']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('The <migration> argument must be an ID or UUID', str(ex))