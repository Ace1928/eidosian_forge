from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import keypairs as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import keypairs
class KeypairsV92TestCase(KeypairsTest):

    def setUp(self):
        super(KeypairsV92TestCase, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.92')

    def test_create_keypair(self):
        name = 'foo'
        key_type = 'some_type'
        public_key = 'fake-public-key'
        kp = self.cs.keypairs.create(name, public_key=public_key, key_type=key_type)
        self.assert_request_id(kp, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/%s' % self.keypair_prefix, body={'keypair': {'name': name, 'public_key': public_key, 'type': key_type}})
        self.assertIsInstance(kp, keypairs.Keypair)

    def test_create_keypair_without_pubkey(self):
        name = 'foo'
        key_type = 'some_type'
        self.assertRaises(TypeError, self.cs.keypairs.create, name, key_type=key_type)