from openstack.clustering.v1 import cluster_attr as ca
from openstack.tests.unit import base
class TestClusterAttr(base.TestCase):

    def setUp(self):
        super(TestClusterAttr, self).setUp()

    def test_basic(self):
        sot = ca.ClusterAttr()
        self.assertEqual('cluster_attributes', sot.resources_key)
        self.assertEqual('/clusters/%(cluster_id)s/attrs/%(path)s', sot.base_path)
        self.assertTrue(sot.allow_list)

    def test_instantiate(self):
        sot = ca.ClusterAttr(**FAKE)
        self.assertEqual(FAKE['cluster_id'], sot.cluster_id)
        self.assertEqual(FAKE['path'], sot.path)
        self.assertEqual(FAKE['id'], sot.node_id)
        self.assertEqual(FAKE['value'], sot.attr_value)