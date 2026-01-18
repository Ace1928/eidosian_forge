import random
import time
import unittest
class ExpiringLRUCacheTests(LRUCacheTests):

    def _getTargetClass(self):
        from repoze.lru import ExpiringLRUCache
        return ExpiringLRUCache

    def _makeOne(self, size, default_timeout=None):
        if default_timeout is None:
            return self._getTargetClass()(size)
        else:
            return self._getTargetClass()(size, default_timeout=default_timeout)

    def check_cache_is_consistent(self, cache):
        self.assertTrue(cache.hand < len(cache.clock_keys))
        self.assertTrue(cache.hand >= 0)
        self.assertEqual(cache.maxpos, cache.size - 1)
        self.assertEqual(len(cache.clock_keys), cache.size)
        self.assertEqual(len(cache.clock_keys), len(cache.clock_refs))
        self.assertTrue(len(cache.data) <= len(cache.clock_refs))
        for key, value in cache.data.items():
            pos, val, timeout = value
            self.assertTrue(type(pos) == type(42) or type(pos) == type(2 ** 128))
            self.assertTrue(pos >= 0)
            self.assertTrue(pos <= cache.maxpos)
            clock_key = cache.clock_keys[pos]
            self.assertTrue(clock_key is key)
            clock_ref = cache.clock_refs[pos]
            self.assertTrue(type(timeout) == type(3.141))
        for clock_ref in cache.clock_refs:
            self.assertTrue(clock_ref is True or clock_ref is False)

    def test_it(self):
        cache = self._makeOne(3)
        self.assertIsNone(cache.get('a'))
        cache.put('a', '1')
        pos, value, expires = cache.data.get('a')
        self.assertEqual(cache.clock_refs[pos], True)
        self.assertEqual(cache.clock_keys[pos], 'a')
        self.assertEqual(value, '1')
        self.assertEqual(cache.get('a'), '1')
        self.assertEqual(cache.hand, pos + 1)
        pos, value, expires = cache.data.get('a')
        self.assertEqual(cache.clock_refs[pos], True)
        self.assertEqual(cache.hand, pos + 1)
        self.assertEqual(len(cache.data), 1)
        cache.put('b', '2')
        pos, value, expires = cache.data.get('b')
        self.assertEqual(cache.clock_refs[pos], True)
        self.assertEqual(cache.clock_keys[pos], 'b')
        self.assertEqual(len(cache.data), 2)
        cache.put('c', '3')
        pos, value, expires = cache.data.get('c')
        self.assertEqual(cache.clock_refs[pos], True)
        self.assertEqual(cache.clock_keys[pos], 'c')
        self.assertEqual(len(cache.data), 3)
        pos, value, expires = cache.data.get('a')
        self.assertEqual(cache.clock_refs[pos], True)
        cache.get('a')
        cache.put('d', '4')
        self.assertEqual(len(cache.data), 3)
        self.assertIsNone(cache.data.get('a'))
        cache.put('e', '5')
        self.assertEqual(len(cache.data), 3)
        self.assertIsNone(cache.data.get('b'))
        self.assertEqual(cache.get('d'), '4')
        self.assertEqual(cache.get('e'), '5')
        self.assertIsNone(cache.get('a'))
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('c'), '3')
        self.check_cache_is_consistent(cache)

    def test_default_timeout(self):
        cache = self._makeOne(3)
        cache.put('foo', 'bar')
        time.sleep(0.1)
        cache.put('FOO', 'BAR')
        self.assertEqual(cache.get('foo'), 'bar')
        self.assertEqual(cache.get('FOO'), 'BAR')
        self.check_cache_is_consistent(cache)
        cache = self._makeOne(3, default_timeout=0.1)
        cache.put('foo', 'bar')
        time.sleep(0.1)
        cache.put('FOO', 'BAR')
        self.assertIsNone(cache.get('foo'))
        self.assertEqual(cache.get('FOO'), 'BAR')
        self.check_cache_is_consistent(cache)

    def test_different_timeouts(self):
        cache = self._makeOne(3, default_timeout=0.1)
        cache.put('one', 1)
        cache.put('two', 2, timeout=0.2)
        cache.put('three', 3, timeout=0.3)
        self.assertEqual(cache.get('one'), 1)
        self.assertEqual(cache.get('two'), 2)
        self.assertEqual(cache.get('three'), 3)
        time.sleep(0.1)
        self.assertIsNone(cache.get('one'))
        self.assertEqual(cache.get('two'), 2)
        self.assertEqual(cache.get('three'), 3)
        time.sleep(0.1)
        self.assertIsNone(cache.get('one'))
        self.assertIsNone(cache.get('two'))
        self.assertEqual(cache.get('three'), 3)
        time.sleep(0.1)
        self.assertIsNone(cache.get('one'))
        self.assertIsNone(cache.get('two'))
        self.assertIsNone(cache.get('three'))
        self.check_cache_is_consistent(cache)

    def test_renew_timeout(self):
        cache = self._makeOne(3, default_timeout=0.2)
        cache.put('foo', 'bar')
        cache.put('foo2', 'bar2', timeout=10)
        cache.put('foo3', 'bar3', timeout=10)
        time.sleep(0.1)
        self.assertEqual(cache.get('foo'), 'bar')
        self.assertEqual(cache.get('foo2'), 'bar2')
        self.assertEqual(cache.get('foo3'), 'bar3')
        self.check_cache_is_consistent(cache)
        cache.put('foo', 'bar')
        cache.put('foo2', 'bar2', timeout=0.1)
        cache.put('foo3', 'bar3')
        time.sleep(0.1)
        self.assertEqual(cache.get('foo'), 'bar')
        self.assertIsNone(cache.get('foo2'))
        self.assertEqual(cache.get('foo3'), 'bar3')
        self.check_cache_is_consistent(cache)