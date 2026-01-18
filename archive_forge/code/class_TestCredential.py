from openstack.identity.v3 import credential
from openstack.tests.unit import base
class TestCredential(base.TestCase):

    def test_basic(self):
        sot = credential.Credential()
        self.assertEqual('credential', sot.resource_key)
        self.assertEqual('credentials', sot.resources_key)
        self.assertEqual('/credentials', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)
        self.assertDictEqual({'type': 'type', 'user_id': 'user_id', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = credential.Credential(**EXAMPLE)
        self.assertEqual(EXAMPLE['blob'], sot.blob)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['user_id'], sot.user_id)