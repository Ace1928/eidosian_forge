import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
class OIDCClientCredentialsTests(BaseOIDCTests, utils.TestCase):

    def setUp(self):
        super(OIDCClientCredentialsTests, self).setUp()
        self.GRANT_TYPE = 'client_credentials'
        self.plugin = oidc.OidcClientCredentials(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, access_token_endpoint=self.ACCESS_TOKEN_ENDPOINT, project_name=self.PROJECT_NAME)

    def test_initial_call_to_get_access_token(self):
        """Test initial call, expect JSON access token."""
        self.requests_mock.post(self.ACCESS_TOKEN_ENDPOINT, json=oidc_fixtures.ACCESS_TOKEN_VIA_PASSWORD_RESP)
        scope = 'profile email'
        payload = {'grant_type': self.GRANT_TYPE, 'scope': scope}
        self.plugin._get_access_token(self.session, payload)
        last_req = self.requests_mock.last_request
        self.assertEqual(self.ACCESS_TOKEN_ENDPOINT, last_req.url)
        self.assertEqual('POST', last_req.method)
        encoded_payload = urllib.parse.urlencode(payload)
        self.assertEqual(encoded_payload, last_req.body)

    def test_second_call_to_protected_url(self):
        """Test subsequent call, expect Keystone token."""
        self.requests_mock.post(self.FEDERATION_AUTH_URL, json=oidc_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': KEYSTONE_TOKEN_VALUE})
        res = self.plugin._get_keystone_token(self.session, self.ACCESS_TOKEN)
        self.assertEqual(self.FEDERATION_AUTH_URL, res.request.url)
        self.assertEqual('POST', res.request.method)
        headers = {'Authorization': 'Bearer ' + self.ACCESS_TOKEN}
        self.assertEqual(headers['Authorization'], res.request.headers['Authorization'])

    def test_end_to_end_workflow(self):
        """Test full OpenID Connect workflow."""
        self.requests_mock.post(self.ACCESS_TOKEN_ENDPOINT, json=oidc_fixtures.ACCESS_TOKEN_VIA_PASSWORD_RESP)
        self.requests_mock.post(self.FEDERATION_AUTH_URL, json=oidc_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': KEYSTONE_TOKEN_VALUE})
        response = self.plugin.get_unscoped_auth_ref(self.session)
        self.assertEqual(KEYSTONE_TOKEN_VALUE, response.auth_token)