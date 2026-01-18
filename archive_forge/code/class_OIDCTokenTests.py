import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
class OIDCTokenTests(utils.TestCase):

    def setUp(self):
        super(OIDCTokenTests, self).setUp()
        self.session = session.Session()
        self.AUTH_URL = 'http://keystone:5000/v3'
        self.IDENTITY_PROVIDER = 'bluepages'
        self.PROTOCOL = 'oidc'
        self.PROJECT_NAME = 'foo project'
        self.ACCESS_TOKEN = uuid.uuid4().hex
        self.FEDERATION_AUTH_URL = '%s/%s' % (self.AUTH_URL, 'OS-FEDERATION/identity_providers/bluepages/protocols/oidc/auth')
        self.plugin = oidc.OidcAccessToken(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, access_token=self.ACCESS_TOKEN, project_name=self.PROJECT_NAME)

    def test_end_to_end_workflow(self):
        """Test full OpenID Connect workflow."""
        self.requests_mock.post(self.FEDERATION_AUTH_URL, json=oidc_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': KEYSTONE_TOKEN_VALUE})
        response = self.plugin.get_unscoped_auth_ref(self.session)
        self.assertEqual(KEYSTONE_TOKEN_VALUE, response.auth_token)