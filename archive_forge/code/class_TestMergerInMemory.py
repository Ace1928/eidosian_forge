import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
class TestMergerInMemory(TestMergerBase):

    def test_cache_trees_with_revision_ids_None(self):
        merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
        original_cache = dict(merger._cached_trees)
        merger.cache_trees_with_revision_ids([None])
        self.assertEqual(original_cache, merger._cached_trees)

    def test_cache_trees_with_revision_ids_no_revision_id(self):
        merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
        original_cache = dict(merger._cached_trees)
        tree = self.make_branch_and_memory_tree('tree')
        merger.cache_trees_with_revision_ids([tree])
        self.assertEqual(original_cache, merger._cached_trees)

    def test_cache_trees_with_revision_ids_having_revision_id(self):
        merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
        original_cache = dict(merger._cached_trees)
        tree = merger.this_branch.repository.revision_tree(b'B-id')
        original_cache[b'B-id'] = tree
        merger.cache_trees_with_revision_ids([tree])
        self.assertEqual(original_cache, merger._cached_trees)

    def test_find_base(self):
        merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
        self.assertEqual(b'A-id', merger.base_rev_id)
        self.assertFalse(merger._is_criss_cross)
        self.assertIs(None, merger._lca_trees)

    def test_find_base_criss_cross(self):
        builder = self.setup_criss_cross_graph()
        merger = self.make_Merger(builder, b'E-id')
        self.assertEqual(b'A-id', merger.base_rev_id)
        self.assertTrue(merger._is_criss_cross)
        self.assertEqual([b'B-id', b'C-id'], [t.get_revision_id() for t in merger._lca_trees])
        builder.build_snapshot([b'E-id'], [], revision_id=b'F-id')
        merger = self.make_Merger(builder, b'D-id')
        self.assertEqual([b'C-id', b'B-id'], [t.get_revision_id() for t in merger._lca_trees])

    def test_find_base_triple_criss_cross(self):
        builder = self.setup_criss_cross_graph()
        builder.build_snapshot([b'A-id'], [], revision_id=b'F-id')
        builder.build_snapshot([b'E-id', b'F-id'], [], revision_id=b'H-id')
        builder.build_snapshot([b'D-id', b'F-id'], [], revision_id=b'G-id')
        merger = self.make_Merger(builder, b'H-id')
        self.assertEqual([b'B-id', b'C-id', b'F-id'], [t.get_revision_id() for t in merger._lca_trees])

    def test_find_base_new_root_criss_cross(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', None, 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([], [('add', ('', None, 'directory', None))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id', b'B-id'], [], revision_id=b'D-id')
        builder.build_snapshot([b'A-id', b'B-id'], [], revision_id=b'C-id')
        merger = self.make_Merger(builder, b'D-id')
        self.assertEqual(b'A-id', merger.base_rev_id)
        self.assertTrue(merger._is_criss_cross)
        self.assertEqual([b'A-id', b'B-id'], [t.get_revision_id() for t in merger._lca_trees])

    def test_no_criss_cross_passed_to_merge_type(self):

        class LCATreesMerger(LoggingMerger):
            supports_lca_trees = True
        merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
        merger.merge_type = LCATreesMerger
        merge_obj = merger.make_merger()
        self.assertIsInstance(merge_obj, LCATreesMerger)
        self.assertFalse('lca_trees' in merge_obj.kwargs)

    def test_criss_cross_passed_to_merge_type(self):
        merger = self.make_Merger(self.setup_criss_cross_graph(), b'E-id')
        merger.merge_type = _mod_merge.Merge3Merger
        merge_obj = merger.make_merger()
        self.assertEqual([b'B-id', b'C-id'], [t.get_revision_id() for t in merger._lca_trees])

    def test_criss_cross_not_supported_merge_type(self):
        merger = self.make_Merger(self.setup_criss_cross_graph(), b'E-id')
        merger.merge_type = LoggingMerger
        merge_obj = merger.make_merger()
        self.assertIsInstance(merge_obj, LoggingMerger)
        self.assertFalse('lca_trees' in merge_obj.kwargs)

    def test_criss_cross_unsupported_merge_type(self):

        class UnsupportedLCATreesMerger(LoggingMerger):
            supports_lca_trees = False
        merger = self.make_Merger(self.setup_criss_cross_graph(), b'E-id')
        merger.merge_type = UnsupportedLCATreesMerger
        merge_obj = merger.make_merger()
        self.assertIsInstance(merge_obj, UnsupportedLCATreesMerger)
        self.assertFalse('lca_trees' in merge_obj.kwargs)