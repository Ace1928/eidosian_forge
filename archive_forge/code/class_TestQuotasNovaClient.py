from novaclient.tests.functional import base
class TestQuotasNovaClient(base.ClientTestBase):
    """Nova quotas functional tests."""
    COMPUTE_API_VERSION = '2.1'
    _quota_resources = ['instances', 'cores', 'ram', 'floating_ips', 'fixed_ips', 'metadata_items', 'injected_files', 'injected_file_content_bytes', 'injected_file_path_bytes', 'key_pairs', 'security_groups', 'security_group_rules', 'server_groups', 'server_group_members']

    def test_quotas_update(self):
        tenant_id = self._get_project_id(self.cli_clients.tenant_name)
        self.addCleanup(self.client.quotas.delete, tenant_id)
        original_quotas = self.client.quotas.get(tenant_id)
        difference = 10
        params = [tenant_id]
        for quota_name in self._quota_resources:
            params.append('--%(name)s %(value)s' % {'name': quota_name.replace('_', '-'), 'value': getattr(original_quotas, quota_name) + difference})
        self.nova('quota-update', params=' '.join(params))
        updated_quotas = self.client.quotas.get(tenant_id)
        for quota_name in self._quota_resources:
            self.assertEqual(getattr(original_quotas, quota_name), getattr(updated_quotas, quota_name) - difference)