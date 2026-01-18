from openstack.identity.v2 import tenant
from openstack.tests.unit import base
class TestTenant(base.TestCase):

    def test_basic(self):
        sot = tenant.Tenant()
        self.assertEqual('tenant', sot.resource_key)
        self.assertEqual('tenants', sot.resources_key)
        self.assertEqual('/tenants', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = tenant.Tenant(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertTrue(sot.is_enabled)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)