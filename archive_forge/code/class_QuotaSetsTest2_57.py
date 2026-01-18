from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import quotas as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class QuotaSetsTest2_57(QuotaSetsTest):
    """Tests the quotas API binding using the 2.57 microversion."""
    data_fixture_class = data.V2_57
    invalid_resources = ['floating_ips', 'fixed_ips', 'networks', 'security_groups', 'security_group_rules', 'injected_files', 'injected_file_content_bytes', 'injected_file_path_bytes']

    def setUp(self):
        super(QuotaSetsTest2_57, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.57')

    def test_tenant_quotas_get(self):
        q = super(QuotaSetsTest2_57, self).test_tenant_quotas_get()
        for invalid_resource in self.invalid_resources:
            self.assertFalse(hasattr(q, invalid_resource), '%s should not be in %s' % (invalid_resource, q))

    def test_force_update_quota(self):
        q = super(QuotaSetsTest2_57, self).test_force_update_quota()
        for invalid_resource in self.invalid_resources:
            self.assertFalse(hasattr(q, invalid_resource), '%s should not be in %s' % (invalid_resource, q))

    def test_update_quota_invalid_resources(self):
        """Tests trying to update quota values for invalid resources."""
        q = self.cs.quotas.get('test')
        self.assertRaises(TypeError, q.update, floating_ips=1)
        self.assertRaises(TypeError, q.update, fixed_ips=1)
        self.assertRaises(TypeError, q.update, security_groups=1)
        self.assertRaises(TypeError, q.update, security_group_rules=1)
        self.assertRaises(TypeError, q.update, networks=1)
        self.assertRaises(TypeError, q.update, injected_files=1)
        self.assertRaises(TypeError, q.update, injected_file_content_bytes=1)
        self.assertRaises(TypeError, q.update, injected_file_path_bytes=1)