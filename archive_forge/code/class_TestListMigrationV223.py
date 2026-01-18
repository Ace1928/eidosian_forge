from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestListMigrationV223(TestListMigration):
    """Test fetch all migrations."""
    MIGRATION_COLUMNS = ['Id', 'Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Old Flavor', 'New Flavor', 'Type', 'Created At', 'Updated At']
    MIGRATION_FIELDS = ['id', 'source_node', 'dest_node', 'source_compute', 'dest_compute', 'dest_host', 'status', 'server_id', 'old_flavor_id', 'new_flavor_id', 'migration_type', 'created_at', 'updated_at']

    def setUp(self):
        super().setUp()
        self._set_mock_microversion('2.23')

    def test_server_migration_list(self):
        arglist = ['--status', 'migrating']
        verifylist = [('status', 'migrating')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'status': 'migrating'}
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))