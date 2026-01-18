from openstack.object_store.v1 import account
from openstack.tests.unit import base
class TestAccount(base.TestCase):

    def setUp(self):
        super(TestAccount, self).setUp()
        self.endpoint = self.cloud.object_store.get_endpoint() + '/'

    def test_basic(self):
        sot = account.Account(**ACCOUNT_EXAMPLE)
        self.assertIsNone(sot.resources_key)
        self.assertIsNone(sot.id)
        self.assertEqual('/', sot.base_path)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_head)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)
        self.assertFalse(sot.allow_create)

    def test_make_it(self):
        sot = account.Account(**ACCOUNT_EXAMPLE)
        self.assertIsNone(sot.id)
        self.assertEqual(int(ACCOUNT_EXAMPLE['x-account-bytes-used']), sot.account_bytes_used)
        self.assertEqual(int(ACCOUNT_EXAMPLE['x-account-container-count']), sot.account_container_count)
        self.assertEqual(int(ACCOUNT_EXAMPLE['x-account-object-count']), sot.account_object_count)
        self.assertEqual(ACCOUNT_EXAMPLE['x-timestamp'], sot.timestamp)

    def test_set_temp_url_key(self):
        sot = account.Account()
        key = 'super-secure-key'
        self.register_uris([dict(method='POST', uri=self.endpoint, status_code=204, validate=dict(headers={'x-account-meta-temp-url-key': key})), dict(method='HEAD', uri=self.endpoint, headers={'x-account-meta-temp-url-key': key})])
        sot.set_temp_url_key(self.cloud.object_store, key)
        self.assert_calls()

    def test_set_account_temp_url_key_second(self):
        sot = account.Account()
        key = 'super-secure-key'
        self.register_uris([dict(method='POST', uri=self.endpoint, status_code=204, validate=dict(headers={'x-account-meta-temp-url-key-2': key})), dict(method='HEAD', uri=self.endpoint, headers={'x-account-meta-temp-url-key-2': key})])
        sot.set_temp_url_key(self.cloud.object_store, key, secondary=True)
        self.assert_calls()