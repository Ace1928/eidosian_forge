import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
class OIDCAuthorizationGrantTests(BaseOIDCTests, utils.TestCase):

    def setUp(self):
        super(OIDCAuthorizationGrantTests, self).setUp()
        self.GRANT_TYPE = 'authorization_code'
        self.plugin = oidc.OidcAuthorizationCode(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, access_token_endpoint=self.ACCESS_TOKEN_ENDPOINT, redirect_uri=self.REDIRECT_URL, project_name=self.PROJECT_NAME, code=self.CODE)

    def test_initial_call_to_get_access_token(self):
        """Test initial call, expect JSON access token."""
        self.requests_mock.post(self.ACCESS_TOKEN_ENDPOINT, json=oidc_fixtures.ACCESS_TOKEN_VIA_AUTH_GRANT_RESP)
        grant_type = 'authorization_code'
        payload = {'grant_type': grant_type, 'redirect_uri': self.REDIRECT_URL, 'code': self.CODE}
        self.plugin._get_access_token(self.session, payload)
        last_req = self.requests_mock.last_request
        self.assertEqual(self.ACCESS_TOKEN_ENDPOINT, last_req.url)
        self.assertEqual('POST', last_req.method)
        encoded_payload = urllib.parse.urlencode(payload)
        self.assertEqual(encoded_payload, last_req.body)