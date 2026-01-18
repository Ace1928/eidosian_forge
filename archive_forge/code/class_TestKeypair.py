from openstack.compute.v2 import keypair
from openstack.tests.unit import base
class TestKeypair(base.TestCase):

    def test_basic(self):
        sot = keypair.Keypair()
        self.assertEqual('keypair', sot.resource_key)
        self.assertEqual('keypairs', sot.resources_key)
        self.assertEqual('/os-keypairs', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'user_id': 'user_id'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = keypair.Keypair(**EXAMPLE)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['deleted'], sot.is_deleted)
        self.assertEqual(EXAMPLE['fingerprint'], sot.fingerprint)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['public_key'], sot.public_key)
        self.assertEqual(EXAMPLE['private_key'], sot.private_key)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['user_id'], sot.user_id)

    def test_make_it_defaults(self):
        EXAMPLE_DEFAULT = EXAMPLE.copy()
        EXAMPLE_DEFAULT.pop('type')
        sot = keypair.Keypair(**EXAMPLE_DEFAULT)
        self.assertEqual(EXAMPLE['type'], sot.type)