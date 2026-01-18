from unittest import mock
from openstack.clustering.v1 import node
from openstack.tests.unit import base
class TestNodeDetail(base.TestCase):

    def test_basic(self):
        sot = node.NodeDetail()
        self.assertEqual('/nodes/%(node_id)s?show_details=True', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)