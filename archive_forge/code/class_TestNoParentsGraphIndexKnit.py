import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
class TestNoParentsGraphIndexKnit(KnitTests):
    """Tests for knits using _KnitGraphIndex with no parents."""

    def make_g_index(self, name, ref_lists=0, nodes=[]):
        builder = GraphIndexBuilder(ref_lists)
        for node, references in nodes:
            builder.add_node(node, references)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file(name, stream)
        return GraphIndex(trans, name, size)

    def test_add_good_unvalidated_index(self):
        unvalidated = self.make_g_index('unvalidated')
        combined = CombinedGraphIndex([unvalidated])
        index = _KnitGraphIndex(combined, lambda: True, parents=False)
        index.scan_unvalidated_index(unvalidated)
        self.assertEqual(frozenset(), index.get_missing_compression_parents())

    def test_parents_deltas_incompatible(self):
        index = CombinedGraphIndex([])
        self.assertRaises(knit.KnitError, _KnitGraphIndex, lambda: True, index, deltas=True, parents=False)

    def two_graph_index(self, catch_adds=False):
        """Build a two-graph index.

        :param deltas: If true, use underlying indices with two node-ref
            lists and 'parent' set to a delta-compressed against tail.
        """
        index1 = self.make_g_index('1', 0, [((b'tip',), b'N0 100'), ((b'tail',), b'')])
        index2 = self.make_g_index('2', 0, [((b'parent',), b' 100 78'), ((b'separate',), b'')])
        combined_index = CombinedGraphIndex([index1, index2])
        if catch_adds:
            self.combined_index = combined_index
            self.caught_entries = []
            add_callback = self.catch_add
        else:
            add_callback = None
        return _KnitGraphIndex(combined_index, lambda: True, parents=False, add_callback=add_callback)

    def test_keys(self):
        index = self.two_graph_index()
        self.assertEqual({(b'tail',), (b'tip',), (b'parent',), (b'separate',)}, set(index.keys()))

    def test_get_position(self):
        index = self.two_graph_index()
        self.assertEqual((index._graph_index._indices[0], 0, 100), index.get_position((b'tip',)))
        self.assertEqual((index._graph_index._indices[1], 100, 78), index.get_position((b'parent',)))

    def test_get_method(self):
        index = self.two_graph_index()
        self.assertEqual('fulltext', index.get_method((b'tip',)))
        self.assertEqual([b'fulltext'], index.get_options((b'parent',)))

    def test_get_options(self):
        index = self.two_graph_index()
        self.assertEqual([b'fulltext', b'no-eol'], index.get_options((b'tip',)))
        self.assertEqual([b'fulltext'], index.get_options((b'parent',)))

    def test_get_parent_map(self):
        index = self.two_graph_index()
        self.assertEqual({(b'parent',): None}, index.get_parent_map([(b'parent',), (b'ghost',)]))

    def catch_add(self, entries):
        self.caught_entries.append(entries)

    def test_add_no_callback_errors(self):
        index = self.two_graph_index()
        self.assertRaises(errors.ReadOnlyError, index.add_records, [((b'new',), b'fulltext,no-eol', (None, 50, 60), [(b'separate',)])])

    def test_add_version_smoke(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), [])])
        self.assertEqual([[((b'new',), b'N50 60')]], self.caught_entries)

    def test_add_version_delta_not_delta_index(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'new',), b'no-eol,line-delta', (None, 0, 100), [])])
        self.assertEqual([], self.caught_entries)

    def test_add_version_same_dup(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 100), [])])
        index.add_records([((b'tip',), b'no-eol,fulltext', (None, 0, 100), [])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 50, 100), [])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 1000), [])])
        self.assertEqual([[], [], [], []], self.caught_entries)

    def test_add_version_different_dup(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'no-eol,line-delta', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'line-delta,no-eol', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext,no-eol', (None, 0, 100), [(b'parent',)])])
        self.assertEqual([], self.caught_entries)

    def test_add_versions(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), []), ((b'new2',), b'fulltext', (None, 0, 6), [])])
        self.assertEqual([((b'new',), b'N50 60'), ((b'new2',), b' 0 6')], sorted(self.caught_entries[0]))
        self.assertEqual(1, len(self.caught_entries))

    def test_add_versions_delta_not_delta_index(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'new',), b'no-eol,line-delta', (None, 0, 100), [(b'parent',)])])
        self.assertEqual([], self.caught_entries)

    def test_add_versions_parents_not_parents_index(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'new',), b'no-eol,fulltext', (None, 0, 100), [(b'parent',)])])
        self.assertEqual([], self.caught_entries)

    def test_add_versions_random_id_accepted(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([], random_id=True)

    def test_add_versions_same_dup(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 100), [])])
        index.add_records([((b'tip',), b'no-eol,fulltext', (None, 0, 100), [])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 50, 100), [])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 1000), [])])
        self.assertEqual([[], [], [], []], self.caught_entries)

    def test_add_versions_different_dup(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'no-eol,line-delta', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'line-delta,no-eol', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext,no-eol', (None, 0, 100), [(b'parent',)])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext,no-eol', (None, 0, 100), []), ((b'tip',), b'no-eol,line-delta', (None, 0, 100), [])])
        self.assertEqual([], self.caught_entries)