from openstack.instance_ha.v1 import host
from openstack.tests.unit import base
class TestHost(base.TestCase):

    def test_basic(self):
        sot = host.Host(HOST)
        self.assertEqual('host', sot.resource_key)
        self.assertEqual('hosts', sot.resources_key)
        self.assertEqual('/segments/%(segment_id)s/hosts', sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertDictEqual({'failover_segment_id': 'failover_segment_id', 'limit': 'limit', 'marker': 'marker', 'on_maintenance': 'on_maintenance', 'reserved': 'reserved', 'sort_dir': 'sort_dir', 'sort_key': 'sort_key', 'type': 'type'}, sot._query_mapping._mapping)

    def test_create(self):
        sot = host.Host(**HOST)
        self.assertEqual(HOST['id'], sot.id)
        self.assertEqual(HOST['uuid'], sot.uuid)
        self.assertEqual(HOST['segment_id'], sot.segment_id)
        self.assertEqual(HOST['created_at'], sot.created_at)
        self.assertEqual(HOST['updated_at'], sot.updated_at)
        self.assertEqual(HOST['name'], sot.name)
        self.assertEqual(HOST['type'], sot.type)
        self.assertEqual(HOST['control_attributes'], sot.control_attributes)
        self.assertEqual(HOST['on_maintenance'], sot.on_maintenance)
        self.assertEqual(HOST['reserved'], sot.reserved)
        self.assertEqual(HOST['failover_segment_id'], sot.failover_segment_id)