import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
class BaseUserPluginTests(object):

    def configure_middleware(self, auth_type, **kwargs):
        opts = loading.get_auth_plugin_conf_options(auth_type)
        self.cfg.register_opts(opts, group=_base.AUTHTOKEN_GROUP)
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(group=_base.AUTHTOKEN_GROUP, auth_type=auth_type, **kwargs)

    def assertTokenDataEqual(self, token_id, token, token_data):
        self.assertEqual(token_id, token_data.auth_token)
        self.assertEqual(token.user_id, token_data.user_id)
        try:
            trust_id = token.trust_id
        except KeyError:
            trust_id = None
        self.assertEqual(trust_id, token_data.trust_id)
        self.assertEqual(self.get_role_names(token), token_data.role_names)

    def get_plugin(self, token_id, service_token_id=None):
        headers = {'X-Auth-Token': token_id}
        if service_token_id:
            headers['X-Service-Token'] = service_token_id
        m = self.create_simple_middleware()
        resp = self.call(m, headers=headers)
        return resp.request.environ['keystone.token_auth']

    def test_user_information(self):
        token_id, token = self.get_token()
        plugin = self.get_plugin(token_id)
        self.assertTokenDataEqual(token_id, token, plugin.user)
        self.assertFalse(plugin.has_service_token)
        self.assertIsNone(plugin.service)

    def test_with_service_information(self):
        token_id, token = self.get_token()
        service_id, service = self.get_token(service=True)
        plugin = self.get_plugin(token_id, service_id)
        self.assertTokenDataEqual(token_id, token, plugin.user)
        self.assertTokenDataEqual(service_id, service, plugin.service)