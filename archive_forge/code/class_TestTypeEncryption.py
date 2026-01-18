from openstack.block_storage.v3 import type
from openstack.tests.unit import base
class TestTypeEncryption(base.TestCase):

    def test_basic(self):
        sot = type.TypeEncryption(**TYPE_ENC)
        self.assertEqual('encryption', sot.resource_key)
        self.assertEqual('encryption', sot.resources_key)
        self.assertEqual('/types/%(volume_type_id)s/encryption', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_delete)
        self.assertFalse(sot.allow_list)
        self.assertTrue(sot.allow_commit)

    def test_new(self):
        sot = type.TypeEncryption.new(encryption_id=FAKE_ID)
        self.assertEqual(FAKE_ID, sot.encryption_id)

    def test_create(self):
        sot = type.TypeEncryption(**TYPE_ENC)
        self.assertEqual(TYPE_ENC['volume_type_id'], sot.volume_type_id)
        self.assertEqual(TYPE_ENC['encryption_id'], sot.encryption_id)
        self.assertEqual(TYPE_ENC['key_size'], sot.key_size)
        self.assertEqual(TYPE_ENC['provider'], sot.provider)
        self.assertEqual(TYPE_ENC['control_location'], sot.control_location)
        self.assertEqual(TYPE_ENC['cipher'], sot.cipher)
        self.assertEqual(TYPE_ENC['deleted'], sot.deleted)
        self.assertEqual(TYPE_ENC['created_at'], sot.created_at)
        self.assertEqual(TYPE_ENC['updated_at'], sot.updated_at)
        self.assertEqual(TYPE_ENC['deleted_at'], sot.deleted_at)