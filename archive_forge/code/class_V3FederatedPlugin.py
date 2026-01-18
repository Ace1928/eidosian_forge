import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
class V3FederatedPlugin(utils.TestCase):
    AUTH_URL = 'http://keystone/v3'

    def setUp(self):
        super(V3FederatedPlugin, self).setUp()
        self.unscoped_token = fixture.V3Token()
        self.unscoped_token_id = uuid.uuid4().hex
        self.scoped_token = copy.deepcopy(self.unscoped_token)
        self.scoped_token.set_project_scope()
        self.scoped_token.methods.append('token')
        self.scoped_token_id = uuid.uuid4().hex
        s = self.scoped_token.add_service('compute', name='nova')
        s.add_standard_endpoints(public='http://nova/public', admin='http://nova/admin', internal='http://nova/internal')
        self.idp = uuid.uuid4().hex
        self.protocol = uuid.uuid4().hex
        self.token_url = '%s/OS-FEDERATION/identity_providers/%s/protocols/%s/auth' % (self.AUTH_URL, self.idp, self.protocol)
        headers = {'X-Subject-Token': self.unscoped_token_id}
        self.unscoped_mock = self.requests_mock.post(self.token_url, json=self.unscoped_token, headers=headers)
        headers = {'X-Subject-Token': self.scoped_token_id}
        auth_url = self.AUTH_URL + '/auth/tokens'
        self.scoped_mock = self.requests_mock.post(auth_url, json=self.scoped_token, headers=headers)

    def get_plugin(self, **kwargs):
        kwargs.setdefault('auth_url', self.AUTH_URL)
        kwargs.setdefault('protocol', self.protocol)
        kwargs.setdefault('identity_provider', self.idp)
        return TesterFederationPlugin(**kwargs)

    def test_federated_url(self):
        plugin = self.get_plugin()
        self.assertEqual(self.token_url, plugin.federated_token_url)

    def test_federated_url_unversioned(self):
        plugin = self.get_plugin(auth_url='http://keystone/')
        self.assertEqual(self.token_url, plugin.federated_token_url)

    def test_unscoped_behaviour(self):
        sess = session.Session(auth=self.get_plugin())
        self.assertEqual(self.unscoped_token_id, sess.get_token())
        self.assertTrue(self.unscoped_mock.called)
        self.assertFalse(self.scoped_mock.called)

    def test_scoped_behaviour(self):
        auth = self.get_plugin(project_id=self.scoped_token.project_id)
        sess = session.Session(auth=auth)
        self.assertEqual(self.scoped_token_id, sess.get_token())
        self.assertTrue(self.unscoped_mock.called)
        self.assertTrue(self.scoped_mock.called)