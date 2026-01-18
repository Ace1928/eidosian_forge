from openstack.identity.v3 import endpoint
from openstack.tests.unit import base
class TestEndpoint(base.TestCase):

    def test_basic(self):
        sot = endpoint.Endpoint()
        self.assertEqual('endpoint', sot.resource_key)
        self.assertEqual('endpoints', sot.resources_key)
        self.assertEqual('/endpoints', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)
        self.assertDictEqual({'interface': 'interface', 'service_id': 'service_id', 'region_id': 'region_id', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = endpoint.Endpoint(**EXAMPLE)
        self.assertTrue(sot.is_enabled)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['interface'], sot.interface)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['region_id'], sot.region_id)
        self.assertEqual(EXAMPLE['service_id'], sot.service_id)
        self.assertEqual(EXAMPLE['url'], sot.url)