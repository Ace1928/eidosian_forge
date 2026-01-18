import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
class TestLiveMemcache(base.BaseAuthTokenTestCase):

    def setUp(self):
        super(TestLiveMemcache, self).setUp()
        global MEMCACHED_AVAILABLE
        if MEMCACHED_AVAILABLE is None:
            try:
                import memcache
                c = memcache.Client(MEMCACHED_SERVERS)
                c.set('ping', 'pong', time=1)
                MEMCACHED_AVAILABLE = c.get('ping') == 'pong'
            except ImportError:
                MEMCACHED_AVAILABLE = False
        if not MEMCACHED_AVAILABLE:
            self.skipTest('memcached not available')

    def test_encrypt_cache_data(self):
        conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'encrypt', 'memcache_secret_key': 'mysecret'}
        token = uuid.uuid4().hex.encode()
        data = uuid.uuid4().hex
        token_cache = self.create_simple_middleware(conf=conf)._token_cache
        token_cache.initialize({})
        token_cache.set(token, data)
        self.assertEqual(token_cache.get(token), data)

    @mock.patch('keystonemiddleware.auth_token._memcache_crypt.unprotect_data')
    def test_corrupted_cache_data(self, mocked_decrypt_data):
        mocked_decrypt_data.side_effect = Exception('corrupted')
        conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'encrypt', 'memcache_secret_key': 'mysecret'}
        token = uuid.uuid4().hex.encode()
        data = uuid.uuid4().hex
        token_cache = self.create_simple_middleware(conf=conf)._token_cache
        token_cache.initialize({})
        token_cache.set(token, data)
        self.assertIsNone(token_cache.get(token))

    def test_sign_cache_data(self):
        conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'mac', 'memcache_secret_key': 'mysecret'}
        token = uuid.uuid4().hex.encode()
        data = uuid.uuid4().hex
        token_cache = self.create_simple_middleware(conf=conf)._token_cache
        token_cache.initialize({})
        token_cache.set(token, data)
        self.assertEqual(token_cache.get(token), data)

    def test_no_memcache_protection(self):
        conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_secret_key': 'mysecret'}
        token = uuid.uuid4().hex.encode()
        data = uuid.uuid4().hex
        token_cache = self.create_simple_middleware(conf=conf)._token_cache
        token_cache.initialize({})
        token_cache.set(token, data)
        self.assertEqual(token_cache.get(token), data)

    def test_memcache_pool(self):
        conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_use_advanced_pool': True}
        token = uuid.uuid4().hex.encode()
        data = uuid.uuid4().hex
        token_cache = self.create_simple_middleware(conf=conf)._token_cache
        token_cache.initialize({})
        token_cache.set(token, data)
        self.assertEqual(token_cache.get(token), data)