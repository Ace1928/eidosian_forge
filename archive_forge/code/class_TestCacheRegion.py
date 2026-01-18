import uuid
from dogpile.cache import api as dogpile
from dogpile.cache.backends import memory
from oslo_config import fixture as config_fixture
from keystone.common import cache
import keystone.conf
from keystone.tests import unit
class TestCacheRegion(unit.BaseTestCase):

    def setUp(self):
        super(TestCacheRegion, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='cache', backend='dogpile.cache.memory')
        cache.CACHE_INVALIDATION_REGION.configure(backend='dogpile.cache.memory', expiration_time=None, replace_existing_backend=True)
        self.region_name = uuid.uuid4().hex
        self.region0 = cache.create_region('test_region')
        self.region1 = cache.create_region('test_region')
        cache.configure_cache(region=self.region0)
        cache.configure_cache(region=self.region1)
        self.cache_dict = {}
        self.backend = memory.MemoryBackend({'cache_dict': self.cache_dict})
        self.region0.backend = self.backend
        self.region1.backend = self.backend

    def _assert_has_no_value(self, values):
        for value in values:
            self.assertIsInstance(value, dogpile.NoValue)

    def test_singular_methods_when_invalidating_the_region(self):
        key = uuid.uuid4().hex
        value = uuid.uuid4().hex
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)
        self.region0.set(key, value)
        self.assertEqual(value, self.region0.get(key))
        self.region1.invalidate()
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)

    def test_region_singular_methods_delete(self):
        key = uuid.uuid4().hex
        value = uuid.uuid4().hex
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)
        self.region0.set(key, value)
        self.assertEqual(value, self.region0.get(key))
        self.region1.delete(key)
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)

    def test_multi_methods_when_invalidating_the_region(self):
        mapping = {uuid.uuid4().hex: uuid.uuid4().hex for _ in range(4)}
        keys = list(mapping.keys())
        values = [mapping[k] for k in keys]
        self._assert_has_no_value(self.region0.get_multi(keys))
        self.region0.set_multi(mapping)
        self.assertEqual(values, self.region0.get_multi(keys))
        self.assertEqual(mapping[keys[0]], self.region0.get(keys[0]))
        self.region1.invalidate()
        self._assert_has_no_value(self.region0.get_multi(keys))

    def test_region_multi_methods_delete(self):
        mapping = {uuid.uuid4().hex: uuid.uuid4().hex for _ in range(4)}
        keys = list(mapping.keys())
        values = [mapping[k] for k in keys]
        self._assert_has_no_value(self.region0.get_multi(keys))
        self.region0.set_multi(mapping)
        keys = list(mapping.keys())
        self.assertEqual(values, self.region0.get_multi(keys))
        self.assertEqual(mapping[keys[0]], self.region0.get(keys[0]))
        self.region1.delete_multi(mapping.keys())
        self._assert_has_no_value(self.region0.get_multi(keys))

    def test_memoize_decorator_when_invalidating_the_region(self):
        memoize = cache.get_memoization_decorator('cache', region=self.region0)

        @memoize
        def func(value):
            return value + uuid.uuid4().hex
        key = uuid.uuid4().hex
        return_value = func(key)
        self.assertEqual(return_value, func(key))
        self.region1.invalidate()
        new_value = func(key)
        self.assertNotEqual(return_value, new_value)

    def test_combination(self):
        memoize = cache.get_memoization_decorator('cache', region=self.region0)

        @memoize
        def func(value):
            return value + uuid.uuid4().hex
        key = uuid.uuid4().hex
        simple_value = uuid.uuid4().hex
        return_value = func(key)
        self.assertEqual(return_value, func(key))
        self.region0.set(key, simple_value)
        self.assertEqual(simple_value, self.region0.get(key))
        self.region1.invalidate()
        new_value = func(key)
        self.assertNotEqual(return_value, new_value)
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)

    def test_direct_region_key_invalidation(self):
        """Invalidate by manually clearing the region key's value.

        NOTE(dstanek): I normally don't like tests that repeat application
        logic, but in this case we need to. There are too many ways that
        the tests above can erroneosly pass that we need this sanity check.
        """
        region_key = cache.RegionInvalidationManager(None, self.region0.name)._region_key
        key = uuid.uuid4().hex
        value = uuid.uuid4().hex
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)
        self.region0.set(key, value)
        self.assertEqual(value, self.region0.get(key))
        cache.CACHE_INVALIDATION_REGION.delete(region_key)
        self.assertIsInstance(self.region0.get(key), dogpile.NoValue)