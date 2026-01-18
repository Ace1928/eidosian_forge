import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestTraits(base.BaseBaremetalTest):
    min_microversion = '1.37'

    def setUp(self):
        super(TestTraits, self).setUp()
        self.node = self.create_node()

    def test_add_remove_node_trait(self):
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual([], node.traits)
        self.conn.baremetal.add_node_trait(self.node, 'CUSTOM_FAKE')
        self.assertEqual(['CUSTOM_FAKE'], self.node.traits)
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual(['CUSTOM_FAKE'], node.traits)
        self.conn.baremetal.add_node_trait(self.node, 'CUSTOM_REAL')
        self.assertEqual(sorted(['CUSTOM_FAKE', 'CUSTOM_REAL']), sorted(self.node.traits))
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual(sorted(['CUSTOM_FAKE', 'CUSTOM_REAL']), sorted(node.traits))
        self.conn.baremetal.remove_node_trait(node, 'CUSTOM_FAKE', ignore_missing=False)
        self.assertEqual(['CUSTOM_REAL'], self.node.traits)
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual(['CUSTOM_REAL'], node.traits)

    def test_set_node_traits(self):
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual([], node.traits)
        traits1 = ['CUSTOM_FAKE', 'CUSTOM_REAL']
        traits2 = ['CUSTOM_FOOBAR']
        self.conn.baremetal.set_node_traits(self.node, traits1)
        self.assertEqual(sorted(traits1), sorted(self.node.traits))
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual(sorted(traits1), sorted(node.traits))
        self.conn.baremetal.set_node_traits(self.node, traits2)
        self.assertEqual(['CUSTOM_FOOBAR'], self.node.traits)
        node = self.conn.baremetal.get_node(self.node)
        self.assertEqual(['CUSTOM_FOOBAR'], node.traits)