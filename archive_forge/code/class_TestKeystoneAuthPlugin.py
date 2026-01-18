import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
class TestKeystoneAuthPlugin(utils.BaseTestCase):
    """Test that the Keystone auth plugin works properly"""

    def setUp(self):
        super(TestKeystoneAuthPlugin, self).setUp()

    def test_get_plugin_from_strategy_keystone(self):
        strategy = auth.get_plugin_from_strategy('keystone')
        self.assertIsInstance(strategy, auth.KeystoneStrategy)
        self.assertTrue(strategy.configure_via_auth)

    def test_get_plugin_from_strategy_keystone_configure_via_auth_false(self):
        strategy = auth.get_plugin_from_strategy('keystone', configure_via_auth=False)
        self.assertIsInstance(strategy, auth.KeystoneStrategy)
        self.assertFalse(strategy.configure_via_auth)

    def test_required_creds(self):
        """
        Test that plugin created without required
        credential pieces raises an exception
        """
        bad_creds = [{}, {'username': 'user1', 'strategy': 'keystone', 'password': 'pass'}, {'password': 'pass', 'strategy': 'keystone', 'auth_url': 'http://localhost/v1'}, {'username': 'user1', 'strategy': 'keystone', 'auth_url': 'http://localhost/v1'}, {'username': 'user1', 'password': 'pass', 'auth_url': 'http://localhost/v1'}, {'username': 'user1', 'password': 'pass', 'strategy': 'keystone', 'auth_url': 'http://localhost/v2.0/'}, {'username': None, 'password': 'pass', 'auth_url': 'http://localhost/v2.0/'}, {'username': 'user1', 'password': 'pass', 'auth_url': 'http://localhost/v2.0/', 'tenant': None}]
        for creds in bad_creds:
            try:
                plugin = auth.KeystoneStrategy(creds)
                plugin.authenticate()
                self.fail('Failed to raise correct exception when supplying bad credentials: %r' % creds)
            except exception.MissingCredentialError:
                continue

    def test_invalid_auth_url_v1(self):
        """
        Test that a 400 during authenticate raises exception.AuthBadRequest
        """

        def fake_do_request(*args, **kwargs):
            resp = webob.Response()
            resp.status = http.BAD_REQUEST
            return (FakeResponse(resp), '')
        self.mock_object(auth.KeystoneStrategy, '_do_request', fake_do_request)
        bad_creds = {'username': 'user1', 'auth_url': 'http://localhost/badauthurl/', 'password': 'pass', 'strategy': 'keystone', 'region': 'RegionOne'}
        plugin = auth.KeystoneStrategy(bad_creds)
        self.assertRaises(exception.AuthBadRequest, plugin.authenticate)

    def test_invalid_auth_url_v2(self):
        """
        Test that a 400 during authenticate raises exception.AuthBadRequest
        """

        def fake_do_request(*args, **kwargs):
            resp = webob.Response()
            resp.status = http.BAD_REQUEST
            return (FakeResponse(resp), '')
        self.mock_object(auth.KeystoneStrategy, '_do_request', fake_do_request)
        bad_creds = {'username': 'user1', 'auth_url': 'http://localhost/badauthurl/v2.0/', 'password': 'pass', 'tenant': 'tenant1', 'strategy': 'keystone', 'region': 'RegionOne'}
        plugin = auth.KeystoneStrategy(bad_creds)
        self.assertRaises(exception.AuthBadRequest, plugin.authenticate)

    def test_v1_auth(self):
        """Test v1 auth code paths"""

        def fake_do_request(cls, url, method, headers=None, body=None):
            if url.find('2.0') != -1:
                self.fail('Invalid v1.0 token path (%s)' % url)
            headers = headers or {}
            resp = webob.Response()
            if headers.get('X-Auth-User') != 'user1' or headers.get('X-Auth-Key') != 'pass':
                resp.status = http.UNAUTHORIZED
            else:
                resp.status = http.OK
                resp.headers.update({'x-image-management-url': 'example.com'})
            return (FakeResponse(resp), '')
        self.mock_object(auth.KeystoneStrategy, '_do_request', fake_do_request)
        unauthorized_creds = [{'username': 'wronguser', 'auth_url': 'http://localhost/badauthurl/', 'strategy': 'keystone', 'region': 'RegionOne', 'password': 'pass'}, {'username': 'user1', 'auth_url': 'http://localhost/badauthurl/', 'strategy': 'keystone', 'region': 'RegionOne', 'password': 'badpass'}]
        for creds in unauthorized_creds:
            try:
                plugin = auth.KeystoneStrategy(creds)
                plugin.authenticate()
                self.fail('Failed to raise NotAuthenticated when supplying bad credentials: %r' % creds)
            except exception.NotAuthenticated:
                continue
        no_strategy_creds = {'username': 'user1', 'auth_url': 'http://localhost/redirect/', 'password': 'pass', 'region': 'RegionOne'}
        try:
            plugin = auth.KeystoneStrategy(no_strategy_creds)
            plugin.authenticate()
            self.fail('Failed to raise MissingCredentialError when supplying no strategy: %r' % no_strategy_creds)
        except exception.MissingCredentialError:
            pass
        good_creds = [{'username': 'user1', 'auth_url': 'http://localhost/redirect/', 'password': 'pass', 'strategy': 'keystone', 'region': 'RegionOne'}]
        for creds in good_creds:
            plugin = auth.KeystoneStrategy(creds)
            self.assertIsNone(plugin.authenticate())
            self.assertEqual('example.com', plugin.management_url)
        for creds in good_creds:
            plugin = auth.KeystoneStrategy(creds, configure_via_auth=False)
            self.assertIsNone(plugin.authenticate())
            self.assertIsNone(plugin.management_url)

    def test_v2_auth(self):
        """Test v2 auth code paths"""
        mock_token = None

        def fake_do_request(cls, url, method, headers=None, body=None):
            if not url.rstrip('/').endswith('v2.0/tokens') or url.count('2.0') != 1:
                self.fail('Invalid v2.0 token path (%s)' % url)
            creds = jsonutils.loads(body)['auth']
            username = creds['passwordCredentials']['username']
            password = creds['passwordCredentials']['password']
            tenant = creds['tenantName']
            resp = webob.Response()
            if username != 'user1' or password != 'pass' or tenant != 'tenant-ok':
                resp.status = http.UNAUTHORIZED
            else:
                resp.status = http.OK
                body = mock_token.token
            return (FakeResponse(resp), jsonutils.dumps(body))
        mock_token = V2Token()
        mock_token.add_service('image', ['RegionOne'])
        self.mock_object(auth.KeystoneStrategy, '_do_request', fake_do_request)
        unauthorized_creds = [{'username': 'wronguser', 'auth_url': 'http://localhost/v2.0', 'password': 'pass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionOne'}, {'username': 'user1', 'auth_url': 'http://localhost/v2.0', 'password': 'badpass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionOne'}, {'username': 'user1', 'auth_url': 'http://localhost/v2.0', 'password': 'pass', 'tenant': 'carterhayes', 'strategy': 'keystone', 'region': 'RegionOne'}]
        for creds in unauthorized_creds:
            try:
                plugin = auth.KeystoneStrategy(creds)
                plugin.authenticate()
                self.fail('Failed to raise NotAuthenticated when supplying bad credentials: %r' % creds)
            except exception.NotAuthenticated:
                continue
        no_region_creds = {'username': 'user1', 'tenant': 'tenant-ok', 'auth_url': 'http://localhost/redirect/v2.0/', 'password': 'pass', 'strategy': 'keystone'}
        plugin = auth.KeystoneStrategy(no_region_creds)
        self.assertIsNone(plugin.authenticate())
        self.assertEqual('http://localhost:9292', plugin.management_url)
        mock_token.add_service('image', ['RegionTwo'])
        try:
            plugin = auth.KeystoneStrategy(no_region_creds)
            plugin.authenticate()
            self.fail('Failed to raise RegionAmbiguity when no region present and multiple regions exist: %r' % no_region_creds)
        except exception.RegionAmbiguity:
            pass
        wrong_region_creds = {'username': 'user1', 'tenant': 'tenant-ok', 'auth_url': 'http://localhost/redirect/v2.0/', 'password': 'pass', 'strategy': 'keystone', 'region': 'NonExistentRegion'}
        try:
            plugin = auth.KeystoneStrategy(wrong_region_creds)
            plugin.authenticate()
            self.fail('Failed to raise NoServiceEndpoint when supplying wrong region: %r' % wrong_region_creds)
        except exception.NoServiceEndpoint:
            pass
        no_strategy_creds = {'username': 'user1', 'tenant': 'tenant-ok', 'auth_url': 'http://localhost/redirect/v2.0/', 'password': 'pass', 'region': 'RegionOne'}
        try:
            plugin = auth.KeystoneStrategy(no_strategy_creds)
            plugin.authenticate()
            self.fail('Failed to raise MissingCredentialError when supplying no strategy: %r' % no_strategy_creds)
        except exception.MissingCredentialError:
            pass
        bad_strategy_creds = {'username': 'user1', 'tenant': 'tenant-ok', 'auth_url': 'http://localhost/redirect/v2.0/', 'password': 'pass', 'region': 'RegionOne', 'strategy': 'keypebble'}
        try:
            plugin = auth.KeystoneStrategy(bad_strategy_creds)
            plugin.authenticate()
            self.fail('Failed to raise BadAuthStrategy when supplying bad auth strategy: %r' % bad_strategy_creds)
        except exception.BadAuthStrategy:
            pass
        mock_token = V2Token()
        mock_token.add_service('image', ['RegionOne', 'RegionTwo'])
        good_creds = [{'username': 'user1', 'auth_url': 'http://localhost/v2.0/', 'password': 'pass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionOne'}, {'username': 'user1', 'auth_url': 'http://localhost/v2.0', 'password': 'pass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionOne'}, {'username': 'user1', 'auth_url': 'http://localhost/v2.0', 'password': 'pass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionTwo'}]
        for creds in good_creds:
            plugin = auth.KeystoneStrategy(creds)
            self.assertIsNone(plugin.authenticate())
            self.assertEqual('http://localhost:9292', plugin.management_url)
        ambiguous_region_creds = {'username': 'user1', 'auth_url': 'http://localhost/v2.0/', 'password': 'pass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionOne'}
        mock_token = V2Token()
        mock_token.add_service('image', ['RegionOne'])
        mock_token.add_service('image', ['RegionOne'])
        try:
            plugin = auth.KeystoneStrategy(ambiguous_region_creds)
            plugin.authenticate()
            self.fail('Failed to raise RegionAmbiguity when non-unique regions exist: %r' % ambiguous_region_creds)
        except exception.RegionAmbiguity:
            pass
        mock_token = V2Token()
        mock_token.add_service('bad-image', ['RegionOne'])
        good_creds = {'username': 'user1', 'auth_url': 'http://localhost/v2.0/', 'password': 'pass', 'tenant': 'tenant-ok', 'strategy': 'keystone', 'region': 'RegionOne'}
        try:
            plugin = auth.KeystoneStrategy(good_creds)
            plugin.authenticate()
            self.fail('Failed to raise NoServiceEndpoint when bad service type encountered')
        except exception.NoServiceEndpoint:
            pass
        mock_token = V2Token()
        mock_token.add_service_no_type()
        try:
            plugin = auth.KeystoneStrategy(good_creds)
            plugin.authenticate()
            self.fail('Failed to raise NoServiceEndpoint when bad service type encountered')
        except exception.NoServiceEndpoint:
            pass
        try:
            plugin = auth.KeystoneStrategy(good_creds, configure_via_auth=False)
            plugin.authenticate()
        except exception.NoServiceEndpoint:
            self.fail('NoServiceEndpoint was raised when authenticate should not check for endpoint.')