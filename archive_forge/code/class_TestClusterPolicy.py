from openstack.clustering.v1 import cluster_policy
from openstack.tests.unit import base
class TestClusterPolicy(base.TestCase):

    def setUp(self):
        super(TestClusterPolicy, self).setUp()

    def test_basic(self):
        sot = cluster_policy.ClusterPolicy()
        self.assertEqual('cluster_policy', sot.resource_key)
        self.assertEqual('cluster_policies', sot.resources_key)
        self.assertEqual('/clusters/%(cluster_id)s/policies', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'policy_name': 'policy_name', 'policy_type': 'policy_type', 'is_enabled': 'enabled', 'sort': 'sort', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_instantiate(self):
        sot = cluster_policy.ClusterPolicy(**FAKE)
        self.assertEqual(FAKE['policy_id'], sot.id)
        self.assertEqual(FAKE['cluster_id'], sot.cluster_id)
        self.assertEqual(FAKE['cluster_name'], sot.cluster_name)
        self.assertEqual(FAKE['data'], sot.data)
        self.assertTrue(sot.is_enabled)
        self.assertEqual(FAKE['policy_id'], sot.policy_id)
        self.assertEqual(FAKE['policy_name'], sot.policy_name)
        self.assertEqual(FAKE['policy_type'], sot.policy_type)