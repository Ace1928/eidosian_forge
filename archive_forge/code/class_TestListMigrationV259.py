from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestListMigrationV259(TestListMigration):
    """Test fetch all migrations."""
    MIGRATION_COLUMNS = ['Id', 'UUID', 'Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Old Flavor', 'New Flavor', 'Type', 'Created At', 'Updated At']
    MIGRATION_FIELDS = ['id', 'uuid', 'source_node', 'dest_node', 'source_compute', 'dest_compute', 'dest_host', 'status', 'server_id', 'old_flavor_id', 'new_flavor_id', 'migration_type', 'created_at', 'updated_at']

    def setUp(self):
        super().setUp()
        self._set_mock_microversion('2.59')

    def test_server_migration_list(self):
        arglist = ['--status', 'migrating', '--limit', '1', '--marker', 'test_kp', '--changes-since', '2019-08-09T08:03:25Z']
        verifylist = [('status', 'migrating'), ('limit', 1), ('marker', 'test_kp'), ('changes_since', '2019-08-09T08:03:25Z')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'status': 'migrating', 'limit': 1, 'paginated': False, 'marker': 'test_kp', 'changes_since': '2019-08-09T08:03:25Z'}
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_migration_list_with_limit_pre_v259(self):
        self._set_mock_microversion('2.58')
        arglist = ['--status', 'migrating', '--limit', '1']
        verifylist = [('status', 'migrating'), ('limit', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.59 or greater is required', str(ex))

    def test_server_migration_list_with_marker_pre_v259(self):
        self._set_mock_microversion('2.58')
        arglist = ['--status', 'migrating', '--marker', 'test_kp']
        verifylist = [('status', 'migrating'), ('marker', 'test_kp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.59 or greater is required', str(ex))

    def test_server_migration_list_with_changes_since_pre_v259(self):
        self._set_mock_microversion('2.58')
        arglist = ['--status', 'migrating', '--changes-since', '2019-08-09T08:03:25Z']
        verifylist = [('status', 'migrating'), ('changes_since', '2019-08-09T08:03:25Z')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.59 or greater is required', str(ex))