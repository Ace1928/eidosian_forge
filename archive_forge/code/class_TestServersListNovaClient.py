import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
class TestServersListNovaClient(base.ClientTestBase):
    """Servers list functional tests."""
    COMPUTE_API_VERSION = '2.1'

    def _create_servers(self, name, number):
        return [self._create_server(name) for i in range(number)]

    def test_list_with_limit(self):
        name = self.name_generate()
        self._create_servers(name, 2)
        output = self.nova('list', params='--limit 1 --name %s' % name)
        servers = output.split('\n')[3:-2]
        self.assertEqual(1, len(servers), output)

    def test_list_with_changes_since(self):
        now = datetime.datetime.isoformat(timeutils.utcnow())
        name = self.name_generate()
        self._create_servers(name, 1)
        output = self.nova('list', params='--changes-since %s' % now)
        self.assertIn(name, output, output)
        now = datetime.datetime.isoformat(timeutils.utcnow())
        output = self.nova('list', params='--changes-since %s' % now)
        self.assertNotIn(name, output, output)

    def test_list_all_servers(self):
        name = self.name_generate()
        precreated_servers = self._create_servers(name, 3)
        output = self.nova('list', params='--limit -1 --name %s' % name)
        for server in precreated_servers:
            self.assertIn(server.id, output)

    def test_list_minimal(self):
        server = self._create_server()
        server_output = self.nova('list --minimal')
        output_uuid = self._get_column_value_from_single_row_table(server_output, 'ID')
        output_name = self._get_column_value_from_single_row_table(server_output, 'Name')
        self.assertEqual(output_uuid, server.id)
        self.assertEqual(output_name, server.name)