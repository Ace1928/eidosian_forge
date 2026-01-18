import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class OpenIDConnectAuthCodeTests(OpenIDConnectBaseTests, utils.TestCase):
    plugin_name = 'v3oidcauthcode'

    def test_options(self):
        options = loading.get_plugin_loader(self.plugin_name).get_options()
        self.assertTrue(set(['redirect-uri', 'code']).issubset(set([o.name for o in options])))

    def test_basic(self):
        access_token_endpoint = uuid.uuid4().hex
        redirect_uri = uuid.uuid4().hex
        authorization_code = uuid.uuid4().hex
        scope = uuid.uuid4().hex
        identity_provider = uuid.uuid4().hex
        protocol = uuid.uuid4().hex
        client_id = uuid.uuid4().hex
        client_secret = uuid.uuid4().hex
        oidc = self.create(code=authorization_code, redirect_uri=redirect_uri, identity_provider=identity_provider, protocol=protocol, access_token_endpoint=access_token_endpoint, client_id=client_id, client_secret=client_secret, scope=scope)
        self.assertEqual(redirect_uri, oidc.redirect_uri)
        self.assertEqual(authorization_code, oidc.code)
        self.assertEqual(scope, oidc.scope)
        self.assertEqual(identity_provider, oidc.identity_provider)
        self.assertEqual(protocol, oidc.protocol)
        self.assertEqual(access_token_endpoint, oidc.access_token_endpoint)
        self.assertEqual(client_id, oidc.client_id)
        self.assertEqual(client_secret, oidc.client_secret)