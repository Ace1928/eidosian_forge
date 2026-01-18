from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import keypairs as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import keypairs
class KeypairsV35TestCase(KeypairsTest):

    def setUp(self):
        super(KeypairsV35TestCase, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.35')

    def test_list_keypairs(self):
        kps = self.cs.keypairs.list(user_id='test_user', marker='test_kp', limit=3)
        self.assert_request_id(kps, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/%s?limit=3&marker=test_kp&user_id=test_user' % self.keypair_prefix)
        for kp in kps:
            self.assertIsInstance(kp, keypairs.Keypair)