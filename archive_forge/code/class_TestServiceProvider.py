from openstack.network.v2 import service_provider
from openstack.tests.unit import base
class TestServiceProvider(base.TestCase):

    def test_basic(self):
        sot = service_provider.ServiceProvider()
        self.assertEqual('service_providers', sot.resources_key)
        self.assertEqual('/service-providers', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = service_provider.ServiceProvider(**EXAMPLE)
        self.assertEqual(EXAMPLE['service_type'], sot.service_type)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['default'], sot.is_default)