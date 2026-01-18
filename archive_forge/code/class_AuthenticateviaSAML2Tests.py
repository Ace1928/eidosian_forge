import base64
import uuid
import requests
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1 import fixture as ksa_fixtures
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
class AuthenticateviaSAML2Tests(utils.TestCase):
    TEST_USER = 'user'
    TEST_PASS = 'pass'
    TEST_IDP = 'tester'
    TEST_PROTOCOL = 'saml2'
    TEST_AUTH_URL = 'http://keystone.test:5000/v3/'
    TEST_IDP_URL = 'https://idp.test'
    TEST_CONSUMER_URL = 'https://openstack4.local/Shibboleth.sso/SAML2/ECP'

    def get_plugin(self, **kwargs):
        kwargs.setdefault('auth_url', self.TEST_AUTH_URL)
        kwargs.setdefault('username', self.TEST_USER)
        kwargs.setdefault('password', self.TEST_PASS)
        kwargs.setdefault('identity_provider', self.TEST_IDP)
        kwargs.setdefault('identity_provider_url', self.TEST_IDP_URL)
        kwargs.setdefault('protocol', self.TEST_PROTOCOL)
        return saml2.V3Saml2Password(**kwargs)

    def sp_url(self, **kwargs):
        kwargs.setdefault('base', self.TEST_AUTH_URL.rstrip('/'))
        kwargs.setdefault('identity_provider', self.TEST_IDP)
        kwargs.setdefault('protocol', self.TEST_PROTOCOL)
        templ = '%(base)s/OS-FEDERATION/identity_providers/%(identity_provider)s/protocols/%(protocol)s/auth'
        return templ % kwargs

    @property
    def calls(self):
        return [r.url.strip('/') for r in self.requests_mock.request_history]

    def basic_header(self, username=TEST_USER, password=TEST_PASS):
        user_pass = ('%s:%s' % (username, password)).encode('utf-8')
        return 'Basic %s' % base64.b64encode(user_pass).decode('utf-8')

    def setUp(self):
        super(AuthenticateviaSAML2Tests, self).setUp()
        self.session = session.Session()
        self.default_sp_url = self.sp_url()

    def test_workflow(self):
        token_id = uuid.uuid4().hex
        token = ksa_fixtures.V3Token()
        self.requests_mock.get(self.default_sp_url, response_list=[dict(headers=CONTENT_TYPE_PAOS_HEADER, content=utils.make_oneline(saml2_fixtures.SP_SOAP_RESPONSE)), dict(headers={'X-Subject-Token': token_id}, json=token)])
        authm = self.requests_mock.post(self.TEST_IDP_URL, content=saml2_fixtures.SAML2_ASSERTION)
        self.requests_mock.post(self.TEST_CONSUMER_URL, status_code=302, headers={'Location': self.sp_url()})
        auth_ref = self.get_plugin().get_auth_ref(self.session)
        self.assertEqual(token_id, auth_ref.auth_token)
        self.assertEqual(self.calls, [self.default_sp_url, self.TEST_IDP_URL, self.TEST_CONSUMER_URL, self.default_sp_url])
        self.assertEqual(self.basic_header(), authm.last_request.headers['Authorization'])
        authn_request = self.requests_mock.request_history[1].text
        self.assertThat(saml2_fixtures.AUTHN_REQUEST, matchers.XMLEquals(authn_request))

    def test_consumer_mismatch_error_workflow(self):
        consumer1 = 'http://keystone.test/Shibboleth.sso/SAML2/ECP'
        consumer2 = 'http://consumer2/Shibboleth.sso/SAML2/ECP'
        soap_response = saml2_fixtures.soap_response(consumer=consumer1)
        saml_assertion = saml2_fixtures.saml_assertion(destination=consumer2)
        self.requests_mock.get(self.default_sp_url, headers=CONTENT_TYPE_PAOS_HEADER, content=soap_response)
        self.requests_mock.post(self.TEST_IDP_URL, content=saml_assertion)
        saml_error = self.requests_mock.post(consumer1)
        self.assertRaises(exceptions.AuthorizationFailure, self.get_plugin().get_auth_ref, self.session)
        self.assertTrue(saml_error.called)

    def test_initial_sp_call_invalid_response(self):
        """Send initial SP HTTP request and receive wrong server response."""
        self.requests_mock.get(self.default_sp_url, headers=CONTENT_TYPE_PAOS_HEADER, text='NON XML RESPONSE')
        self.assertRaises(exceptions.AuthorizationFailure, self.get_plugin().get_auth_ref, self.session)
        self.assertEqual(self.calls, [self.default_sp_url])