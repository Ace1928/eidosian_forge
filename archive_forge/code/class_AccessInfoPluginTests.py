import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.identity import access as access_plugin
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
class AccessInfoPluginTests(utils.TestCase):

    def setUp(self):
        super(AccessInfoPluginTests, self).setUp()
        self.session = session.Session()
        self.auth_token = uuid.uuid4().hex

    def _plugin(self, **kwargs):
        token = fixture.V3Token()
        s = token.add_service('identity')
        s.add_standard_endpoints(public=self.TEST_ROOT_URL)
        auth_ref = access.create(body=token, auth_token=self.auth_token)
        return access_plugin.AccessInfoPlugin(auth_ref, **kwargs)

    def test_auth_ref(self):
        plugin_obj = self._plugin()
        self.assertEqual(self.TEST_ROOT_URL, plugin_obj.get_endpoint(self.session, service_type='identity', interface='public'))
        self.assertEqual(self.auth_token, plugin_obj.get_token(session))

    def test_auth_url(self):
        auth_url = 'http://keystone.test.url'
        obj = self._plugin(auth_url=auth_url)
        self.assertEqual(auth_url, obj.get_endpoint(self.session, interface=plugin.AUTH_INTERFACE))

    def test_invalidate(self):
        plugin = self._plugin()
        auth_ref = plugin.auth_ref
        self.assertIsInstance(auth_ref, access.AccessInfo)
        self.assertFalse(plugin.invalidate())
        self.assertIs(auth_ref, plugin.auth_ref)

    def test_project_auth_properties(self):
        plugin = self._plugin()
        auth_ref = plugin.auth_ref
        self.assertIsNone(auth_ref.project_domain_id)
        self.assertIsNone(auth_ref.project_domain_name)
        self.assertIsNone(auth_ref.project_id)
        self.assertIsNone(auth_ref.project_name)

    def test_domain_auth_properties(self):
        plugin = self._plugin()
        auth_ref = plugin.auth_ref
        self.assertIsNone(auth_ref.domain_id)
        self.assertIsNone(auth_ref.domain_name)