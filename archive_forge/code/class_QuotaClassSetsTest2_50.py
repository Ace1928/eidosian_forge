from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class QuotaClassSetsTest2_50(QuotaClassSetsTest):
    """Tests the quota classes API binding using the 2.50 microversion."""
    api_version = '2.50'
    invalid_resources = ['floating_ips', 'fixed_ips', 'networks', 'security_groups', 'security_group_rules']

    def setUp(self):
        super(QuotaClassSetsTest2_50, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion(self.api_version))

    def test_class_quotas_get(self):
        """Tests that network-related resources aren't in a 2.50 response
        and server group related resources are in the response.
        """
        q = super(QuotaClassSetsTest2_50, self).test_class_quotas_get()
        for invalid_resource in self.invalid_resources:
            self.assertFalse(hasattr(q, invalid_resource), '%s should not be in %s' % (invalid_resource, q))
        for valid_resource in ('server_groups', 'server_group_members'):
            self.assertTrue(hasattr(q, valid_resource), '%s should be in %s' % (invalid_resource, q))

    def test_update_quota(self):
        """Tests that network-related resources aren't in a 2.50 response
        and server group related resources are in the response.
        """
        q = super(QuotaClassSetsTest2_50, self).test_update_quota()
        for invalid_resource in self.invalid_resources:
            self.assertFalse(hasattr(q, invalid_resource), '%s should not be in %s' % (invalid_resource, q))
        for valid_resource in ('server_groups', 'server_group_members'):
            self.assertTrue(hasattr(q, valid_resource), '%s should be in %s' % (invalid_resource, q))

    def test_update_quota_invalid_resources(self):
        """Tests trying to update quota class values for invalid resources.

        This will fail with TypeError because the network-related resource
        kwargs aren't defined.
        """
        q = self.cs.quota_classes.get('test')
        self.assertRaises(TypeError, q.update, floating_ips=1)
        self.assertRaises(TypeError, q.update, fixed_ips=1)
        self.assertRaises(TypeError, q.update, security_groups=1)
        self.assertRaises(TypeError, q.update, security_group_rules=1)
        self.assertRaises(TypeError, q.update, networks=1)
        return q