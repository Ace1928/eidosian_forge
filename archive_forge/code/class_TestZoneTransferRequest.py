from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import base
class TestZoneTransferRequest(base.TestCase):

    def test_basic(self):
        sot = zone_transfer.ZoneTransferRequest()
        self.assertEqual('transfer_requests', sot.resources_key)
        self.assertEqual('/zones/tasks/transfer_requests', sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'status': 'status'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = zone_transfer.ZoneTransferRequest(**EXAMPLE_REQUEST)
        self.assertEqual(IDENTIFIER, sot.id)
        self.assertEqual(EXAMPLE_REQUEST['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE_REQUEST['updated_at'], sot.updated_at)
        self.assertEqual(EXAMPLE_REQUEST['description'], sot.description)
        self.assertEqual(EXAMPLE_REQUEST['key'], sot.key)
        self.assertEqual(EXAMPLE_REQUEST['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE_REQUEST['status'], sot.status)
        self.assertEqual(EXAMPLE_REQUEST['target_project_id'], sot.target_project_id)
        self.assertEqual(EXAMPLE_REQUEST['zone_id'], sot.zone_id)
        self.assertEqual(EXAMPLE_REQUEST['zone_name'], sot.zone_name)