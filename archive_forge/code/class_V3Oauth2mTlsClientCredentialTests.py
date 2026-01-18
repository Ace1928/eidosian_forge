import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class V3Oauth2mTlsClientCredentialTests(utils.TestCase):

    def setUp(self):
        super(V3Oauth2mTlsClientCredentialTests, self).setUp()
        self.auth_url = uuid.uuid4().hex

    def create(self, **kwargs):
        kwargs.setdefault('auth_url', self.auth_url)
        loader = loading.get_plugin_loader('v3oauth2mtlsclientcredential')
        return loader.load_from_options(**kwargs)

    def test_basic(self):
        client_id = uuid.uuid4().hex
        oauth2_endpoint = 'https://localhost/token'
        client_cred = self.create(oauth2_endpoint=oauth2_endpoint, oauth2_client_id=client_id)
        self.assertEqual(self.auth_url, client_cred.auth_url)
        self.assertEqual(client_id, client_cred.oauth2_client_id)
        self.assertEqual(oauth2_endpoint, client_cred.oauth2_endpoint)

    def test_without_oauth2_endpoint(self):
        client_id = uuid.uuid4().hex
        self.assertRaises(exceptions.OptionError, self.create, oauth2_client_id=client_id)

    def test_without_client_id(self):
        oauth2_endpoint = 'https://localhost/token'
        self.assertRaises(exceptions.OptionError, self.create, oauth2_endpoint=oauth2_endpoint, oauth2_client_secret=uuid.uuid4().hex)