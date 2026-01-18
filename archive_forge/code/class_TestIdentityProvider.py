from openstack.identity.v3 import identity_provider
from openstack.tests.unit import base
class TestIdentityProvider(base.TestCase):

    def test_basic(self):
        sot = identity_provider.IdentityProvider()
        self.assertEqual('identity_provider', sot.resource_key)
        self.assertEqual('identity_providers', sot.resources_key)
        self.assertEqual('/OS-FEDERATION/identity_providers', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.create_exclude_id_from_body)
        self.assertEqual('PATCH', sot.commit_method)
        self.assertEqual('PUT', sot.create_method)
        self.assertDictEqual({'id': 'id', 'limit': 'limit', 'marker': 'marker', 'is_enabled': 'enabled'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = identity_provider.IdentityProvider(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['id'], sot.name)
        self.assertEqual(EXAMPLE['domain_id'], sot.domain_id)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['is_enabled'], sot.is_enabled)
        self.assertEqual(EXAMPLE['remote_ids'], sot.remote_ids)