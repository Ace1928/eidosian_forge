import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalVif(base.BaseBaremetalTest):
    min_microversion = '1.28'

    def setUp(self):
        super(TestBareMetalVif, self).setUp()
        self.node = self.create_node(network_interface='noop')
        self.vif_id = '200712fc-fdfb-47da-89a6-2d19f76c7618'

    def test_node_vif_attach_detach(self):
        self.conn.baremetal.attach_vif_to_node(self.node, self.vif_id)
        self.conn.baremetal.list_node_vifs(self.node)
        res = self.conn.baremetal.detach_vif_from_node(self.node, self.vif_id, ignore_missing=False)
        self.assertTrue(res)

    def test_node_vif_negative(self):
        uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.attach_vif_to_node, uuid, self.vif_id)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.list_node_vifs, uuid)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.detach_vif_from_node, uuid, self.vif_id, ignore_missing=False)