import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class OpenIDConnectDeviceAuthorizationTests(utils.TestCase):
    plugin_name = 'v3oidcdeviceauthz'

    def setUp(self):
        super(OpenIDConnectDeviceAuthorizationTests, self).setUp()
        self.auth_url = uuid.uuid4().hex

    def create(self, **kwargs):
        kwargs.setdefault('auth_url', self.auth_url)
        loader = loading.get_plugin_loader(self.plugin_name)
        return loader.load_from_options(**kwargs)

    def test_options(self):
        options = loading.get_plugin_loader(self.plugin_name).get_options()
        self.assertTrue(set(['client-id', 'client-secret', 'access-token-endpoint', 'openid-scope', 'discovery-endpoint', 'device-authorization-endpoint']).issubset(set([o.name for o in options])))

    def test_basic(self):
        access_token_endpoint = uuid.uuid4().hex
        device_authorization_endpoint = uuid.uuid4().hex
        scope = uuid.uuid4().hex
        identity_provider = uuid.uuid4().hex
        protocol = uuid.uuid4().hex
        client_id = uuid.uuid4().hex
        client_secret = uuid.uuid4().hex
        dev_authz_endpt = device_authorization_endpoint
        oidc = self.create(identity_provider=identity_provider, protocol=protocol, access_token_endpoint=access_token_endpoint, device_authorization_endpoint=dev_authz_endpt, client_id=client_id, client_secret=client_secret, scope=scope)
        self.assertEqual(dev_authz_endpt, oidc.device_authorization_endpoint)
        self.assertEqual(identity_provider, oidc.identity_provider)
        self.assertEqual(protocol, oidc.protocol)
        self.assertEqual(access_token_endpoint, oidc.access_token_endpoint)
        self.assertEqual(client_id, oidc.client_id)
        self.assertEqual(client_secret, oidc.client_secret)
        self.assertEqual(scope, oidc.scope)