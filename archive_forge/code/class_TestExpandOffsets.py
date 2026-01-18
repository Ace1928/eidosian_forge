import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
class TestExpandOffsets(tests.TestCase):

    def make_index(self, size, recommended_pages=None):
        """Make an index with a generic size.

        This doesn't actually create anything on disk, it just primes a
        BTreeGraphIndex with the recommended information.
        """
        index = btree_index.BTreeGraphIndex(transport.get_transport_from_url('memory:///'), 'test-index', size=size)
        if recommended_pages is not None:
            index._recommended_pages = recommended_pages
        return index

    def set_cached_offsets(self, index, cached_offsets):
        """Monkeypatch to give a canned answer for _get_offsets_for...()."""

        def _get_offsets_to_cached_pages():
            cached = set(cached_offsets)
            return cached
        index._get_offsets_to_cached_pages = _get_offsets_to_cached_pages

    def prepare_index(self, index, node_ref_lists, key_length, key_count, row_lengths, cached_offsets):
        """Setup the BTreeGraphIndex with some pre-canned information."""
        index.node_ref_lists = node_ref_lists
        index._key_length = key_length
        index._key_count = key_count
        index._row_lengths = row_lengths
        index._compute_row_offsets()
        index._root_node = btree_index._InternalNode(b'internal\noffset=0\n')
        self.set_cached_offsets(index, cached_offsets)

    def make_100_node_index(self):
        index = self.make_index(4096 * 100, 6)
        self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=1000, row_lengths=[1, 99], cached_offsets=[0, 50])
        return index

    def make_1000_node_index(self):
        index = self.make_index(4096 * 1000, 6)
        self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=90000, row_lengths=[1, 9, 990], cached_offsets=[0, 5, 500])
        return index

    def assertNumPages(self, expected_pages, index, size):
        index._size = size
        self.assertEqual(expected_pages, index._compute_total_pages_in_index())

    def assertExpandOffsets(self, expected, index, offsets):
        self.assertEqual(expected, index._expand_offsets(offsets), 'We did not get the expected value after expanding %s' % (offsets,))

    def test_default_recommended_pages(self):
        index = self.make_index(None)
        self.assertEqual(1, index._recommended_pages)

    def test__compute_total_pages_in_index(self):
        index = self.make_index(None)
        self.assertNumPages(1, index, 1024)
        self.assertNumPages(1, index, 4095)
        self.assertNumPages(1, index, 4096)
        self.assertNumPages(2, index, 4097)
        self.assertNumPages(2, index, 8192)
        self.assertNumPages(76, index, 4096 * 75 + 10)

    def test__find_layer_start_and_stop(self):
        index = self.make_1000_node_index()
        self.assertEqual((0, 1), index._find_layer_first_and_end(0))
        self.assertEqual((1, 10), index._find_layer_first_and_end(1))
        self.assertEqual((1, 10), index._find_layer_first_and_end(9))
        self.assertEqual((10, 1000), index._find_layer_first_and_end(10))
        self.assertEqual((10, 1000), index._find_layer_first_and_end(99))
        self.assertEqual((10, 1000), index._find_layer_first_and_end(999))

    def test_unknown_size(self):
        index = self.make_index(None, 10)
        self.assertExpandOffsets([0], index, [0])
        self.assertExpandOffsets([1, 4, 9], index, [1, 4, 9])

    def test_more_than_recommended(self):
        index = self.make_index(4096 * 100, 2)
        self.assertExpandOffsets([1, 10], index, [1, 10])
        self.assertExpandOffsets([1, 10, 20], index, [1, 10, 20])

    def test_read_all_from_root(self):
        index = self.make_index(4096 * 10, 20)
        self.assertExpandOffsets(list(range(10)), index, [0])

    def test_read_all_when_cached(self):
        index = self.make_index(4096 * 10, 5)
        self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=1000, row_lengths=[1, 9], cached_offsets=[0, 1, 2, 5, 6])
        self.assertExpandOffsets([3, 4, 7, 8, 9], index, [3])
        self.assertExpandOffsets([3, 4, 7, 8, 9], index, [8])
        self.assertExpandOffsets([3, 4, 7, 8, 9], index, [9])

    def test_no_root_node(self):
        index = self.make_index(4096 * 10, 5)
        self.assertExpandOffsets([0], index, [0])

    def test_include_neighbors(self):
        index = self.make_100_node_index()
        self.assertExpandOffsets([9, 10, 11, 12, 13, 14, 15], index, [12])
        self.assertExpandOffsets([88, 89, 90, 91, 92, 93, 94], index, [91])
        self.assertExpandOffsets([1, 2, 3, 4, 5, 6], index, [2])
        self.assertExpandOffsets([94, 95, 96, 97, 98, 99], index, [98])
        self.assertExpandOffsets([1, 2, 3, 80, 81, 82], index, [2, 81])
        self.assertExpandOffsets([1, 2, 3, 9, 10, 11, 80, 81, 82], index, [2, 10, 81])

    def test_stop_at_cached(self):
        index = self.make_100_node_index()
        self.set_cached_offsets(index, [0, 10, 19])
        self.assertExpandOffsets([11, 12, 13, 14, 15, 16], index, [11])
        self.assertExpandOffsets([11, 12, 13, 14, 15, 16], index, [12])
        self.assertExpandOffsets([12, 13, 14, 15, 16, 17, 18], index, [15])
        self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [16])
        self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [17])
        self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [18])

    def test_cannot_fully_expand(self):
        index = self.make_100_node_index()
        self.set_cached_offsets(index, [0, 10, 12])
        self.assertExpandOffsets([11], index, [11])

    def test_overlap(self):
        index = self.make_100_node_index()
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [12, 13])
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [11, 14])

    def test_stay_within_layer(self):
        index = self.make_1000_node_index()
        self.assertExpandOffsets([1, 2, 3, 4], index, [2])
        self.assertExpandOffsets([6, 7, 8, 9], index, [6])
        self.assertExpandOffsets([6, 7, 8, 9], index, [9])
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [10])
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15, 16], index, [13])
        self.set_cached_offsets(index, [0, 4, 12])
        self.assertExpandOffsets([5, 6, 7, 8, 9], index, [7])
        self.assertExpandOffsets([10, 11], index, [11])

    def test_small_requests_unexpanded(self):
        index = self.make_100_node_index()
        self.set_cached_offsets(index, [0])
        self.assertExpandOffsets([1], index, [1])
        self.assertExpandOffsets([50], index, [50])
        self.assertExpandOffsets([49, 50, 51, 59, 60, 61], index, [50, 60])
        index = self.make_1000_node_index()
        self.set_cached_offsets(index, [0])
        self.assertExpandOffsets([1], index, [1])
        self.set_cached_offsets(index, [0, 1])
        self.assertExpandOffsets([100], index, [100])
        self.set_cached_offsets(index, [0, 1, 100])
        self.assertExpandOffsets([2, 3, 4, 5, 6, 7], index, [2])
        self.assertExpandOffsets([2, 3, 4, 5, 6, 7], index, [4])
        self.set_cached_offsets(index, [0, 1, 2, 3, 4, 5, 6, 7, 100])
        self.assertExpandOffsets([102, 103, 104, 105, 106, 107, 108], index, [105])