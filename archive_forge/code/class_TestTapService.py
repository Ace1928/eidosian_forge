from openstack.network.v2 import tap_service
from openstack.tests.unit import base
class TestTapService(base.TestCase):

    def test_basic(self):
        sot = tap_service.TapService()
        self.assertEqual('tap_service', sot.resource_key)
        self.assertEqual('tap_services', sot.resources_key)
        self.assertEqual('/taas/tap_services', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = tap_service.TapService(**EXAMPLE)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['port_id'], sot.port_id)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'name': 'name', 'project_id': 'project_id', 'sort_key': 'sort_key', 'sort_dir': 'sort_dir'}, sot._query_mapping._mapping)