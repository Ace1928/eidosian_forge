import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class TestUnencryptedFileCredentialStore(CredentialStoreTestCase):
    """Tests for the UnencryptedFileCredentialStore class."""

    def setUp(self):
        ignore, self.filename = tempfile.mkstemp()
        self.store = UnencryptedFileCredentialStore(self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_save_and_load(self):
        credential = self.make_credential('consumer key')
        self.store.save(credential, 'unique key')
        credential2 = self.store.load('unique key')
        self.assertEqual(credential.consumer.key, credential2.consumer.key)

    def test_unique_id_doesnt_matter(self):
        credential = self.make_credential('consumer key')
        self.store.save(credential, 'some key')
        credential2 = self.store.load('some other key')
        self.assertEqual(credential.consumer.key, credential2.consumer.key)

    def test_file_only_contains_one_credential(self):
        credential1 = self.make_credential('consumer key')
        credential2 = self.make_credential('consumer key2')
        self.store.save(credential1, 'unique key 1')
        self.store.save(credential1, 'unique key 2')
        loaded = self.store.load('unique key 1')
        self.assertEqual(loaded.consumer.key, credential2.consumer.key)