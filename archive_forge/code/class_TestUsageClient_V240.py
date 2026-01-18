from novaclient.tests.functional.v2.legacy import test_usage
class TestUsageClient_V240(test_usage.TestUsageClient):
    COMPUTE_API_VERSION = '2.40'

    def test_get(self):
        start, end = self._create_servers_in_time_window()
        tenant_id = self._get_project_id(self.cli_clients.tenant_name)
        usage = self.client.usage.get(tenant_id, start=start, end=end, limit=1)
        self.assertEqual(tenant_id, usage.tenant_id)
        self.assertEqual(1, len(usage.server_usages))

    def test_list(self):
        start, end = self._create_servers_in_time_window()
        usages = self.client.usage.list(start=start, end=end, detailed=True, limit=1)
        self.assertEqual(1, len(usages))
        self.assertEqual(1, len(usages[0].server_usages))