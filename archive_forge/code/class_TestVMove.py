from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import base
class TestVMove(base.TestCase):

    def test_basic(self):
        sot = vmove.VMove(VMOVE)
        self.assertEqual('vmove', sot.resource_key)
        self.assertEqual('vmoves', sot.resources_key)
        self.assertEqual('/notifications/%(notification_id)s/vmoves', sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_fetch)
        self.assertDictEqual({'status': 'status', 'type': 'type', 'limit': 'limit', 'marker': 'marker', 'sort_dir': 'sort_dir', 'sort_key': 'sort_key'}, sot._query_mapping._mapping)

    def test_create(self):
        sot = vmove.VMove(**VMOVE)
        self.assertEqual(VMOVE['id'], sot.id)
        self.assertEqual(VMOVE['uuid'], sot.uuid)
        self.assertEqual(VMOVE['notification_id'], sot.notification_id)
        self.assertEqual(VMOVE['created_at'], sot.created_at)
        self.assertEqual(VMOVE['updated_at'], sot.updated_at)
        self.assertEqual(VMOVE['server_id'], sot.server_id)
        self.assertEqual(VMOVE['server_name'], sot.server_name)
        self.assertEqual(VMOVE['source_host'], sot.source_host)
        self.assertEqual(VMOVE['dest_host'], sot.dest_host)
        self.assertEqual(VMOVE['start_time'], sot.start_time)
        self.assertEqual(VMOVE['end_time'], sot.end_time)
        self.assertEqual(VMOVE['status'], sot.status)
        self.assertEqual(VMOVE['type'], sot.type)
        self.assertEqual(VMOVE['message'], sot.message)