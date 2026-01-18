import random
import time
import unittest
class DecoratorTests(unittest.TestCase):

    def _getTargetClass(self):
        from repoze.lru import lru_cache
        return lru_cache

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_ctor_no_size(self):
        from repoze.lru import UnboundedCache
        decorator = self._makeOne(maxsize=None)
        self.assertIsInstance(decorator.cache, UnboundedCache)
        self.assertEqual(decorator.cache._data, {})

    def test_ctor_w_size_no_timeout(self):
        from repoze.lru import LRUCache
        decorator = self._makeOne(maxsize=10)
        self.assertIsInstance(decorator.cache, LRUCache)
        self.assertEqual(decorator.cache.size, 10)

    def test_ctor_w_size_w_timeout(self):
        from repoze.lru import ExpiringLRUCache
        decorator = self._makeOne(maxsize=10, timeout=30)
        self.assertIsInstance(decorator.cache, ExpiringLRUCache)
        self.assertEqual(decorator.cache.size, 10)
        self.assertEqual(decorator.cache.default_timeout, 30)

    def test_ctor_nocache(self):
        decorator = self._makeOne(10, None)
        self.assertEqual(decorator.cache.size, 10)

    def test_singlearg(self):
        cache = DummyLRUCache()
        decorator = self._makeOne(0, cache)

        def wrapped(key):
            return key
        decorated = decorator(wrapped)
        result = decorated(1)
        self.assertEqual(cache[1,], 1)
        self.assertEqual(result, 1)
        self.assertEqual(len(cache), 1)
        result = decorated(2)
        self.assertEqual(cache[2,], 2)
        self.assertEqual(result, 2)
        self.assertEqual(len(cache), 2)
        result = decorated(2)
        self.assertEqual(cache[2,], 2)
        self.assertEqual(result, 2)
        self.assertEqual(len(cache), 2)

    def test_cache_attr(self):
        cache = DummyLRUCache()
        decorator = self._makeOne(0, cache)

        def wrapped(key):
            return key
        decorated = decorator(wrapped)
        self.assertTrue(decorated._cache is cache)

    def test_multiargs(self):
        cache = DummyLRUCache()
        decorator = self._makeOne(0, cache)

        def moreargs(*args):
            return args
        decorated = decorator(moreargs)
        result = decorated(3, 4, 5)
        self.assertEqual(cache[3, 4, 5], (3, 4, 5))
        self.assertEqual(result, (3, 4, 5))
        self.assertEqual(len(cache), 1)

    def test_multiargs_keywords(self):
        cache = DummyLRUCache()
        decorator = self._makeOne(0, cache)

        def moreargs(*args, **kwargs):
            return (args, kwargs)
        decorated = decorator(moreargs)
        result = decorated(3, 4, 5, a=1, b=2, c=3)
        self.assertEqual(cache[(3, 4, 5), frozenset([('a', 1), ('b', 2), ('c', 3)])], ((3, 4, 5), {'a': 1, 'b': 2, 'c': 3}))
        self.assertEqual(result, ((3, 4, 5), {'a': 1, 'b': 2, 'c': 3}))
        self.assertEqual(len(cache), 1)

    def test_multiargs_keywords_ignore_unhashable_true(self):
        cache = DummyLRUCache()
        decorator = self._makeOne(0, cache, ignore_unhashable_args=True)

        def moreargs(*args, **kwargs):
            return (args, kwargs)
        decorated = decorator(moreargs)
        result = decorated(3, 4, 5, a=1, b=[1, 2, 3])
        self.assertEqual(len(cache), 0)
        self.assertEqual(result, ((3, 4, 5), {'a': 1, 'b': [1, 2, 3]}))

    def test_multiargs_keywords_ignore_unhashable(self):
        cache = DummyLRUCache()
        decorator = self._makeOne(0, cache, ignore_unhashable_args=False)

        def moreargs(*args, **kwargs):
            return (args, kwargs)
        decorated = decorator(moreargs)
        with self.assertRaises(TypeError):
            decorated(3, 4, 5, a=1, b=[1, 2, 3])

    def test_expiry(self):

        @self._makeOne(1, None, timeout=0.1)
        def sleep_a_bit(param):
            time.sleep(0.1)
            return 2 * param
        start = time.time()
        result1 = sleep_a_bit('hello')
        stop = time.time()
        self.assertEqual(result1, 2 * 'hello')
        self.assertTrue(stop - start > 0.1)
        start = time.time()
        result2 = sleep_a_bit('hello')
        stop = time.time()
        self.assertEqual(result2, 2 * 'hello')
        self.assertTrue(stop - start < 0.1)
        time.sleep(0.1)
        start = time.time()
        result3 = sleep_a_bit('hello')
        stop = time.time()
        self.assertEqual(result3, 2 * 'hello')
        self.assertTrue(stop - start > 0.1)

    def test_partial(self):

        def add(a, b):
            return a + b
        from functools import partial
        from repoze.lru import lru_cache
        add_five = partial(add, 5)
        decorated = lru_cache(20)(add_five)
        self.assertEqual(decorated(3), 8)