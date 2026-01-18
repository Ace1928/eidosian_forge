import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
class TestMemcachePoolAbstraction(utils.TestCase):

    def setUp(self):
        super(TestMemcachePoolAbstraction, self).setUp()
        self.useFixture(fixtures.MockPatch('oslo_cache._memcache_pool._MemcacheClient'))

    def test_abstraction_layer_reserve_places_connection_back_in_pool(self):
        cache_pool = _cache._MemcacheClientPool(memcache_servers=[], arguments={}, maxsize=1, unused_timeout=10)
        conn = None
        with cache_pool.reserve() as client:
            self.assertEqual(cache_pool._pool._acquired, 1)
            conn = client
        self.assertEqual(cache_pool._pool._acquired, 0)
        with cache_pool.reserve() as client:
            self.assertEqual(conn, client)