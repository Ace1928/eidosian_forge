from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(node.Node, 'fetch', lambda self, session: self)
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodeInventory(base.TestCase):

    def setUp(self):
        super().setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion='1.81')

    def test_get_inventory(self):
        node_inventory = {'inventory': {'memory': {'physical_mb': 3072}, 'cpu': {'count': 1, 'model_name': 'qemu64', 'architecture': 'x86_64'}, 'disks': [{'name': 'testvm1.qcow2', 'size': 11811160064}], 'interfaces': [{'mac_address': '52:54:00:c7:02:45'}], 'system_vendor': {'product_name': 'testvm1', 'manufacturer': 'Sushy Emulator'}, 'boot': {'current_boot_mode': 'uefi'}}, 'plugin_data': {'fake_plugin_data'}}
        self.session.get.return_value.json.return_value = node_inventory
        res = self.node.get_node_inventory(self.session, self.node.id)
        self.assertEqual(node_inventory, res)
        self.session.get.assert_called_once_with('nodes/%s/inventory' % self.node.id, headers=mock.ANY, microversion='1.81', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)