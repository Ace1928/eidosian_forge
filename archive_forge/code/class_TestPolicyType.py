from openstack.clustering.v1 import policy_type
from openstack.tests.unit import base
class TestPolicyType(base.TestCase):

    def test_basic(self):
        sot = policy_type.PolicyType()
        self.assertEqual('policy_type', sot.resource_key)
        self.assertEqual('policy_types', sot.resources_key)
        self.assertEqual('/policy-types', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)

    def test_instantiate(self):
        sot = policy_type.PolicyType(**FAKE)
        self.assertEqual(FAKE['name'], sot._get_id(sot))
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['schema'], sot.schema)
        self.assertEqual(FAKE['support_status'], sot.support_status)