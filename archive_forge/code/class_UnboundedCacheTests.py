import random
import time
import unittest
class UnboundedCacheTests(unittest.TestCase):

    def _getTargetClass(self):
        from repoze.lru import UnboundedCache
        return UnboundedCache

    def _makeOne(self):
        return self._getTargetClass()()

    def test_ctor(self):
        cache = self._makeOne()
        self.assertEqual(cache._data, {})

    def test_get_miss_no_default(self):
        cache = self._makeOne()
        self.assertIsNone(cache.get('nonesuch'))

    def test_get_miss_explicit_default(self):
        cache = self._makeOne()
        default = object()
        self.assertIs(cache.get('nonesuch', default), default)

    def test_get_hit(self):
        cache = self._makeOne()
        extant = cache._data['extant'] = object()
        self.assertIs(cache.get('extant'), extant)

    def test_clear(self):
        cache = self._makeOne()
        extant = cache._data['extant'] = object()
        cache.clear()
        self.assertIsNone(cache.get('extant'))

    def test_invalidate_miss(self):
        cache = self._makeOne()
        cache.invalidate('nonesuch')

    def test_invalidate_hit(self):
        cache = self._makeOne()
        extant = cache._data['extant'] = object()
        cache.invalidate('extant')
        self.assertIsNone(cache.get('extant'))

    def test_put(self):
        cache = self._makeOne()
        extant = object()
        cache.put('extant', extant)
        self.assertIs(cache._data['extant'], extant)