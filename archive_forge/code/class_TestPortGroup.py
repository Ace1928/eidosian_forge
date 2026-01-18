from openstack.baremetal.v1 import port_group
from openstack.tests.unit import base
class TestPortGroup(base.TestCase):

    def test_basic(self):
        sot = port_group.PortGroup()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('portgroups', sot.resources_key)
        self.assertEqual('/portgroups', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)

    def test_instantiate(self):
        sot = port_group.PortGroup(**FAKE)
        self.assertEqual(FAKE['uuid'], sot.id)
        self.assertEqual(FAKE['address'], sot.address)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['extra'], sot.extra)
        self.assertEqual(FAKE['internal_info'], sot.internal_info)
        self.assertEqual(FAKE['links'], sot.links)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['node_uuid'], sot.node_id)
        self.assertEqual(FAKE['ports'], sot.ports)
        self.assertEqual(FAKE['standalone_ports_supported'], sot.is_standalone_ports_supported)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)