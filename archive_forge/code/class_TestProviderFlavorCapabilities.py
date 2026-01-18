from openstack.load_balancer.v2 import provider
from openstack.tests.unit import base
class TestProviderFlavorCapabilities(base.TestCase):

    def test_basic(self):
        test_flav_cap = provider.ProviderFlavorCapabilities()
        self.assertEqual('flavor_capabilities', test_flav_cap.resources_key)
        self.assertEqual('/lbaas/providers/%(provider)s/flavor_capabilities', test_flav_cap.base_path)
        self.assertFalse(test_flav_cap.allow_create)
        self.assertFalse(test_flav_cap.allow_fetch)
        self.assertFalse(test_flav_cap.allow_commit)
        self.assertFalse(test_flav_cap.allow_delete)
        self.assertTrue(test_flav_cap.allow_list)

    def test_make_it(self):
        test_flav_cap = provider.ProviderFlavorCapabilities(**EXAMPLE)
        self.assertEqual(EXAMPLE['name'], test_flav_cap.name)
        self.assertEqual(EXAMPLE['description'], test_flav_cap.description)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'name': 'name', 'description': 'description'}, test_flav_cap._query_mapping._mapping)