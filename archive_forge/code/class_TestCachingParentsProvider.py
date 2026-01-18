from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestCachingParentsProvider(tests.TestCase):
    """These tests run with:

    self.inst_pp, a recording parents provider with a graph of a->b, and b is a
    ghost.
    self.caching_pp, a CachingParentsProvider layered on inst_pp.
    """

    def setUp(self):
        super().setUp()
        dict_pp = _mod_graph.DictParentsProvider({b'a': (b'b',)})
        self.inst_pp = InstrumentedParentsProvider(dict_pp)
        self.caching_pp = _mod_graph.CachingParentsProvider(self.inst_pp)

    def test_get_parent_map(self):
        """Requesting the same revision should be returned from cache"""
        self.assertEqual({}, self.caching_pp._cache)
        self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'a']))
        self.assertEqual([b'a'], self.inst_pp.calls)
        self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'a']))
        self.assertEqual([b'a'], self.inst_pp.calls)
        self.assertEqual({b'a': (b'b',)}, self.caching_pp._cache)

    def test_get_parent_map_not_present(self):
        """The cache should also track when a revision doesn't exist"""
        self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
        self.assertEqual([b'b'], self.inst_pp.calls)
        self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
        self.assertEqual([b'b'], self.inst_pp.calls)

    def test_get_parent_map_mixed(self):
        """Anything that can be returned from cache, should be"""
        self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
        self.assertEqual([b'b'], self.inst_pp.calls)
        self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'a', b'b']))
        self.assertEqual([b'b', b'a'], self.inst_pp.calls)

    def test_get_parent_map_repeated(self):
        """Asking for the same parent 2x will only forward 1 request."""
        self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'b', b'a', b'b']))
        self.assertEqual([b'a', b'b'], sorted(self.inst_pp.calls))

    def test_note_missing_key(self):
        """After noting that a key is missing it is cached."""
        self.caching_pp.note_missing_key(b'b')
        self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
        self.assertEqual([], self.inst_pp.calls)
        self.assertEqual({b'b'}, self.caching_pp.missing_keys)

    def test_get_cached_parent_map(self):
        self.assertEqual({}, self.caching_pp.get_cached_parent_map([b'a']))
        self.assertEqual([], self.inst_pp.calls)
        self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'a']))
        self.assertEqual([b'a'], self.inst_pp.calls)
        self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_cached_parent_map([b'a']))