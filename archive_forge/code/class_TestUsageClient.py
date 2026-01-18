import datetime
from novaclient.tests.functional import base
class TestUsageClient(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.1'

    def _create_servers_in_time_window(self):
        start = datetime.datetime.now()
        self._create_server()
        self._create_server()
        end = datetime.datetime.now()
        return (start, end)

    def test_get(self):
        start, end = self._create_servers_in_time_window()
        tenant_id = self._get_project_id(self.cli_clients.tenant_name)
        usage = self.client.usage.get(tenant_id, start=start, end=end)
        self.assertEqual(tenant_id, usage.tenant_id)
        self.assertGreaterEqual(len(usage.server_usages), 2)

    def test_list(self):
        start, end = self._create_servers_in_time_window()
        tenant_id = self._get_project_id(self.cli_clients.tenant_name)
        usages = self.client.usage.list(start=start, end=end, detailed=True)
        tenant_ids = [usage.tenant_id for usage in usages]
        self.assertIn(tenant_id, tenant_ids)
        for usage in usages:
            if usage.tenant_id == tenant_id:
                self.assertGreaterEqual(len(usage.server_usages), 2)