import datetime
from novaclient.tests.functional import base
class TestUsageCLI(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.1'

    def _get_num_servers_from_usage_output(self):
        output = self.nova('usage')
        servers = self._get_column_value_from_single_row_table(output, 'Servers')
        return int(servers)

    def _get_num_servers_by_tenant_from_usage_output(self):
        tenant_id = self._get_project_id(self.cli_clients.tenant_name)
        output = self.nova('usage --tenant=%s' % tenant_id)
        servers = self._get_column_value_from_single_row_table(output, 'Servers')
        return int(servers)

    def test_usage(self):
        before = self._get_num_servers_from_usage_output()
        self._create_server()
        after = self._get_num_servers_from_usage_output()
        self.assertGreater(after, before)

    def test_usage_tenant(self):
        before = self._get_num_servers_by_tenant_from_usage_output()
        self._create_server()
        after = self._get_num_servers_by_tenant_from_usage_output()
        self.assertGreater(after, before)