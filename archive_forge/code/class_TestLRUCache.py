from dulwich import lru_cache
from dulwich.tests import TestCase
class TestLRUCache(TestCase):
    """Test that LRU cache properly keeps track of entries."""

    def test_cache_size(self):
        cache = lru_cache.LRUCache(max_cache=10)
        self.assertEqual(10, cache.cache_size())
        cache = lru_cache.LRUCache(max_cache=256)
        self.assertEqual(256, cache.cache_size())
        cache.resize(512)
        self.assertEqual(512, cache.cache_size())

    def test_missing(self):
        cache = lru_cache.LRUCache(max_cache=10)
        self.assertNotIn('foo', cache)
        self.assertRaises(KeyError, cache.__getitem__, 'foo')
        cache['foo'] = 'bar'
        self.assertEqual('bar', cache['foo'])
        self.assertIn('foo', cache)
        self.assertNotIn('bar', cache)

    def test_map_None(self):
        cache = lru_cache.LRUCache(max_cache=10)
        self.assertNotIn(None, cache)
        cache[None] = 1
        self.assertEqual(1, cache[None])
        cache[None] = 2
        self.assertEqual(2, cache[None])
        cache[1] = 3
        cache[None] = 1
        cache[None]
        cache[1]
        cache[None]
        self.assertEqual([None, 1], [n.key for n in cache._walk_lru()])

    def test_add__null_key(self):
        cache = lru_cache.LRUCache(max_cache=10)
        self.assertRaises(ValueError, cache.add, lru_cache._null_key, 1)

    def test_overflow(self):
        """Adding extra entries will pop out old ones."""
        cache = lru_cache.LRUCache(max_cache=1, after_cleanup_count=1)
        cache['foo'] = 'bar'
        cache['baz'] = 'biz'
        self.assertNotIn('foo', cache)
        self.assertIn('baz', cache)
        self.assertEqual('biz', cache['baz'])

    def test_by_usage(self):
        """Accessing entries bumps them up in priority."""
        cache = lru_cache.LRUCache(max_cache=2)
        cache['baz'] = 'biz'
        cache['foo'] = 'bar'
        self.assertEqual('biz', cache['baz'])
        cache['nub'] = 'in'
        self.assertNotIn('foo', cache)

    def test_cleanup(self):
        """Test that we can use a cleanup function."""
        cleanup_called = []

        def cleanup_func(key, val):
            cleanup_called.append((key, val))
        cache = lru_cache.LRUCache(max_cache=2, after_cleanup_count=2)
        cache.add('baz', '1', cleanup=cleanup_func)
        cache.add('foo', '2', cleanup=cleanup_func)
        cache.add('biz', '3', cleanup=cleanup_func)
        self.assertEqual([('baz', '1')], cleanup_called)
        cache['foo']
        cache.clear()
        self.assertEqual([('baz', '1'), ('biz', '3'), ('foo', '2')], cleanup_called)

    def test_cleanup_on_replace(self):
        """Replacing an object should cleanup the old value."""
        cleanup_called = []

        def cleanup_func(key, val):
            cleanup_called.append((key, val))
        cache = lru_cache.LRUCache(max_cache=2)
        cache.add(1, 10, cleanup=cleanup_func)
        cache.add(2, 20, cleanup=cleanup_func)
        cache.add(2, 25, cleanup=cleanup_func)
        self.assertEqual([(2, 20)], cleanup_called)
        self.assertEqual(25, cache[2])
        cache[2] = 26
        self.assertEqual([(2, 20), (2, 25)], cleanup_called)

    def test_len(self):
        cache = lru_cache.LRUCache(max_cache=10, after_cleanup_count=10)
        cache[1] = 10
        cache[2] = 20
        cache[3] = 30
        cache[4] = 40
        self.assertEqual(4, len(cache))
        cache[5] = 50
        cache[6] = 60
        cache[7] = 70
        cache[8] = 80
        self.assertEqual(8, len(cache))
        cache[1] = 15
        self.assertEqual(8, len(cache))
        cache[9] = 90
        cache[10] = 100
        cache[11] = 110
        self.assertEqual(10, len(cache))
        self.assertEqual([11, 10, 9, 1, 8, 7, 6, 5, 4, 3], [n.key for n in cache._walk_lru()])

    def test_cleanup_shrinks_to_after_clean_count(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=3)
        cache.add(1, 10)
        cache.add(2, 20)
        cache.add(3, 25)
        cache.add(4, 30)
        cache.add(5, 35)
        self.assertEqual(5, len(cache))
        cache.add(6, 40)
        self.assertEqual(3, len(cache))

    def test_after_cleanup_larger_than_max(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=10)
        self.assertEqual(5, cache._after_cleanup_count)

    def test_after_cleanup_none(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=None)
        self.assertEqual(4, cache._after_cleanup_count)

    def test_cleanup_2(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=2)
        cache.add(1, 10)
        cache.add(2, 20)
        cache.add(3, 25)
        cache.add(4, 30)
        cache.add(5, 35)
        self.assertEqual(5, len(cache))
        cache.cleanup()
        self.assertEqual(2, len(cache))

    def test_preserve_last_access_order(self):
        cache = lru_cache.LRUCache(max_cache=5)
        cache.add(1, 10)
        cache.add(2, 20)
        cache.add(3, 25)
        cache.add(4, 30)
        cache.add(5, 35)
        self.assertEqual([5, 4, 3, 2, 1], [n.key for n in cache._walk_lru()])
        cache[2]
        cache[5]
        cache[3]
        cache[2]
        self.assertEqual([2, 3, 5, 4, 1], [n.key for n in cache._walk_lru()])

    def test_get(self):
        cache = lru_cache.LRUCache(max_cache=5)
        cache.add(1, 10)
        cache.add(2, 20)
        self.assertEqual(20, cache.get(2))
        self.assertEqual(None, cache.get(3))
        obj = object()
        self.assertIs(obj, cache.get(3, obj))
        self.assertEqual([2, 1], [n.key for n in cache._walk_lru()])
        self.assertEqual(10, cache.get(1))
        self.assertEqual([1, 2], [n.key for n in cache._walk_lru()])

    def test_keys(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=5)
        cache[1] = 2
        cache[2] = 3
        cache[3] = 4
        self.assertEqual([1, 2, 3], sorted(cache.keys()))
        cache[4] = 5
        cache[5] = 6
        cache[6] = 7
        self.assertEqual([2, 3, 4, 5, 6], sorted(cache.keys()))

    def test_resize_smaller(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=4)
        cache[1] = 2
        cache[2] = 3
        cache[3] = 4
        cache[4] = 5
        cache[5] = 6
        self.assertEqual([1, 2, 3, 4, 5], sorted(cache.keys()))
        cache[6] = 7
        self.assertEqual([3, 4, 5, 6], sorted(cache.keys()))
        cache.resize(max_cache=3, after_cleanup_count=2)
        self.assertEqual([5, 6], sorted(cache.keys()))
        cache[7] = 8
        self.assertEqual([5, 6, 7], sorted(cache.keys()))
        cache[8] = 9
        self.assertEqual([7, 8], sorted(cache.keys()))

    def test_resize_larger(self):
        cache = lru_cache.LRUCache(max_cache=5, after_cleanup_count=4)
        cache[1] = 2
        cache[2] = 3
        cache[3] = 4
        cache[4] = 5
        cache[5] = 6
        self.assertEqual([1, 2, 3, 4, 5], sorted(cache.keys()))
        cache[6] = 7
        self.assertEqual([3, 4, 5, 6], sorted(cache.keys()))
        cache.resize(max_cache=8, after_cleanup_count=6)
        self.assertEqual([3, 4, 5, 6], sorted(cache.keys()))
        cache[7] = 8
        cache[8] = 9
        cache[9] = 10
        cache[10] = 11
        self.assertEqual([3, 4, 5, 6, 7, 8, 9, 10], sorted(cache.keys()))
        cache[11] = 12
        self.assertEqual([6, 7, 8, 9, 10, 11], sorted(cache.keys()))