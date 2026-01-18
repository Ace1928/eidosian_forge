import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
class BaseOIDCTests(object):

    def setUp(self):
        super(BaseOIDCTests, self).setUp()
        self.session = session.Session()
        self.AUTH_URL = 'http://keystone:5000/v3'
        self.IDENTITY_PROVIDER = 'bluepages'
        self.PROTOCOL = 'oidc'
        self.USER_NAME = 'oidc_user@example.com'
        self.PROJECT_NAME = 'foo project'
        self.PASSWORD = uuid.uuid4().hex
        self.CLIENT_ID = uuid.uuid4().hex
        self.CLIENT_SECRET = uuid.uuid4().hex
        self.ACCESS_TOKEN = uuid.uuid4().hex
        self.ACCESS_TOKEN_ENDPOINT = 'https://localhost:8020/oidc/token'
        self.FEDERATION_AUTH_URL = '%s/%s' % (self.AUTH_URL, 'OS-FEDERATION/identity_providers/bluepages/protocols/oidc/auth')
        self.REDIRECT_URL = 'urn:ietf:wg:oauth:2.0:oob'
        self.CODE = '4/M9TNz2G9WVwYxSjx0w9AgA1bOmryJltQvOhQMq0czJs.cnLNVAfqwG'
        self.DISCOVERY_URL = 'https://localhost:8020/oidc/.well-known/openid-configuration'
        self.GRANT_TYPE = None

    def test_grant_type_and_plugin_missmatch(self):
        self.assertRaises(exceptions.OidcGrantTypeMissmatch, self.plugin.__class__, self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, grant_type=uuid.uuid4().hex)

    def test_can_pass_grant_type_but_warning_is_issued(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, grant_type=self.GRANT_TYPE)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert 'grant_type' in str(w[-1].message)

    def test_discovery_not_found(self):
        self.requests_mock.get('http://not.found', status_code=404)
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint='http://not.found')
        self.assertRaises(exceptions.http.NotFound, plugin._get_discovery_document, self.session)

    def test_no_discovery(self):
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, access_token_endpoint=self.ACCESS_TOKEN_ENDPOINT)
        self.assertEqual(self.ACCESS_TOKEN_ENDPOINT, plugin.access_token_endpoint)

    def test_load_discovery(self):
        self.requests_mock.get(self.DISCOVERY_URL, json=oidc_fixtures.DISCOVERY_DOCUMENT)
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL)
        self.assertEqual(oidc_fixtures.DISCOVERY_DOCUMENT['token_endpoint'], plugin._get_access_token_endpoint(self.session))

    def test_no_access_token_endpoint(self):
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET)
        self.assertRaises(exceptions.OidcAccessTokenEndpointNotFound, plugin._get_access_token_endpoint, self.session)

    def test_invalid_discovery_document(self):
        self.requests_mock.get(self.DISCOVERY_URL, json={})
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL)
        self.assertRaises(exceptions.InvalidOidcDiscoveryDocument, plugin._get_discovery_document, self.session)

    def test_load_discovery_override_by_endpoints(self):
        self.requests_mock.get(self.DISCOVERY_URL, json=oidc_fixtures.DISCOVERY_DOCUMENT)
        access_token_endpoint = uuid.uuid4().hex
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL, access_token_endpoint=access_token_endpoint)
        self.assertEqual(access_token_endpoint, plugin._get_access_token_endpoint(self.session))

    def test_wrong_grant_type(self):
        self.requests_mock.get(self.DISCOVERY_URL, json={'grant_types_supported': ['foo', 'bar']})
        plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL)
        self.assertRaises(exceptions.OidcPluginNotSupported, plugin.get_unscoped_auth_ref, self.session)