from .. import fifo_cache, tests
class TestFIFOSizeCache(tests.TestCase):

    def test_add_is_present(self):
        c = fifo_cache.FIFOSizeCache()
        c[1] = '2'
        self.assertTrue(1 in c)
        self.assertEqual(1, len(c))
        self.assertEqual('2', c[1])
        self.assertEqual('2', c.get(1))
        self.assertEqual('2', c.get(1, None))
        self.assertEqual([1], list(c))
        self.assertEqual({1}, c.keys())
        self.assertEqual([(1, '2')], sorted(c.items()))
        self.assertEqual(['2'], sorted(c.values()))
        self.assertEqual({1: '2'}, c)
        self.assertEqual(1024 * 1024, c.cache_size())

    def test_missing(self):
        c = fifo_cache.FIFOSizeCache()
        self.assertRaises(KeyError, c.__getitem__, 1)
        self.assertFalse(1 in c)
        self.assertEqual(0, len(c))
        self.assertEqual(None, c.get(1))
        self.assertEqual(None, c.get(1, None))
        self.assertEqual([], list(c))
        self.assertEqual(set(), c.keys())
        self.assertEqual([], list(c.items()))
        self.assertEqual([], list(c.values()))
        self.assertEqual({}, c)

    def test_add_maintains_fifo(self):
        c = fifo_cache.FIFOSizeCache(10, 8)
        c[1] = 'ab'
        c[2] = 'cde'
        c[3] = 'fghi'
        self.assertEqual({1: 'ab', 2: 'cde', 3: 'fghi'}, c)
        c[4] = 'jkl'
        self.assertEqual({3: 'fghi', 4: 'jkl'}, c)
        c[3] = 'mnop'
        self.assertEqual({3: 'mnop', 4: 'jkl'}, c)
        c[5] = 'qrst'
        self.assertEqual({3: 'mnop', 5: 'qrst'}, c)

    def test_adding_large_key(self):
        c = fifo_cache.FIFOSizeCache(10, 8)
        c[1] = 'abcdefgh'
        self.assertEqual({}, c)
        c[1] = 'abcdefg'
        self.assertEqual({1: 'abcdefg'}, c)
        c[1] = 'abcdefgh'
        self.assertEqual({}, c)
        self.assertEqual(0, c._value_size)

    def test_resize_smaller(self):
        c = fifo_cache.FIFOSizeCache(20, 16)
        c[1] = 'a'
        c[2] = 'bc'
        c[3] = 'def'
        c[4] = 'ghij'
        c.resize(10, 8)
        self.assertEqual({1: 'a', 2: 'bc', 3: 'def', 4: 'ghij'}, c)
        self.assertEqual(10, c.cache_size())
        c[5] = 'k'
        self.assertEqual({3: 'def', 4: 'ghij', 5: 'k'}, c)
        c.resize(5, 4)
        self.assertEqual({5: 'k'}, c)

    def test_resize_larger(self):
        c = fifo_cache.FIFOSizeCache(10, 8)
        c[1] = 'a'
        c[2] = 'bc'
        c[3] = 'def'
        c[4] = 'ghij'
        c.resize(12, 10)
        self.assertEqual({1: 'a', 2: 'bc', 3: 'def', 4: 'ghij'}, c)
        c[5] = 'kl'
        self.assertEqual({1: 'a', 2: 'bc', 3: 'def', 4: 'ghij', 5: 'kl'}, c)
        c[6] = 'mn'
        self.assertEqual({4: 'ghij', 5: 'kl', 6: 'mn'}, c)