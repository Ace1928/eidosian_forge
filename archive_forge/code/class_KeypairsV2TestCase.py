from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import keypairs as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import keypairs
class KeypairsV2TestCase(KeypairsTest):

    def setUp(self):
        super(KeypairsV2TestCase, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.0')

    def test_create_keypair(self):
        name = 'foo'
        kp = self.cs.keypairs.create(name)
        self.assert_request_id(kp, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/%s' % self.keypair_prefix, body={'keypair': {'name': name}})
        self.assertIsInstance(kp, keypairs.Keypair)

    def test_import_keypair(self):
        name = 'foo'
        pub_key = 'fake-public-key'
        kp = self.cs.keypairs.create(name, pub_key)
        self.assert_request_id(kp, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/%s' % self.keypair_prefix, body={'keypair': {'name': name, 'public_key': pub_key}})
        self.assertIsInstance(kp, keypairs.Keypair)