import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
class K2KFederatedProjectTests(utils.TestCase):
    TEST_ROOT_URL = 'http://127.0.0.1:5000/'
    TEST_URL = '%s%s' % (TEST_ROOT_URL, 'v3')
    TEST_PASS = 'password'
    REQUEST_ECP_URL = TEST_URL + '/auth/OS-FEDERATION/saml2/ecp'
    SP_ID = 'sp1'
    SP_ROOT_URL = 'https://example.com/v3'
    SP_URL = 'https://example.com/Shibboleth.sso/SAML2/ECP'
    SP_AUTH_URL = SP_ROOT_URL + '/OS-FEDERATION/identity_providers/testidp/protocols/saml2/auth'

    def setUp(self):
        super(K2KFederatedProjectTests, self).setUp()
        self.token_v3 = fixture.V3Token()
        self.token_v3.add_service_provider(self.SP_ID, self.SP_AUTH_URL, self.SP_URL)
        self.session = session.Session()
        self.collection_key = 'projects'
        self.model = projects.Project
        self.URL = '%s%s' % (self.SP_ROOT_URL, '/auth/projects')
        self.k2kplugin = self.get_plugin()
        self._mock_k2k_flow_urls()

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault('domain_id', uuid.uuid4().hex)
        kwargs.setdefault('enabled', True)
        kwargs.setdefault('name', uuid.uuid4().hex)
        return kwargs

    def _get_base_plugin(self):
        self.stub_url('POST', ['auth', 'tokens'], headers={'X-Subject-Token': uuid.uuid4().hex}, json=self.token_v3)
        return v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)

    def _mock_k2k_flow_urls(self):
        self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
        self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, content=k2k_fixtures.ECP_ENVELOPE.encode(), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=200)
        self.requests_mock.register_uri('POST', self.SP_URL, content=k2k_fixtures.TOKEN_BASED_ECP.encode(), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=302)
        self.requests_mock.register_uri('GET', self.SP_AUTH_URL, json=k2k_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': k2k_fixtures.UNSCOPED_TOKEN_HEADER})

    def get_plugin(self, **kwargs):
        kwargs.setdefault('base_plugin', self._get_base_plugin())
        kwargs.setdefault('service_provider', self.SP_ID)
        return v3.Keystone2Keystone(**kwargs)

    def test_list_projects(self):
        k2k_client = client.Client(session=self.session, auth=self.k2kplugin)
        self.requests_mock.get(self.URL, json={self.collection_key: [self.new_ref(), self.new_ref()]})
        self.requests_mock.get(self.SP_ROOT_URL, json={'version': fixture.discovery.V3Discovery(self.SP_ROOT_URL)})
        returned_list = k2k_client.federation.projects.list()
        self.assertThat(returned_list, matchers.HasLength(2))
        for project in returned_list:
            self.assertIsInstance(project, self.model)