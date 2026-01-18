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
class TestMergerEntriesLCA(TestMergerBase):

    def make_merge_obj(self, builder, other_revision_id, interesting_files=None):
        merger = self.make_Merger(builder, other_revision_id, interesting_files=interesting_files)
        return merger.make_merger()

    def test_simple(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('a', b'a\nb\nC\nc\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'A-id'], [('modify', ('a', b'a\nB\nb\nc\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('a', b'a\nB\nb\nC\nc\nE\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('modify', ('a', b'a\nB\nb\nC\nc\n'))], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        self.assertEqual([b'B-id', b'C-id'], [t.get_revision_id() for t in merge_obj._lca_trees])
        self.assertEqual(b'A-id', merge_obj.base_tree.get_revision_id())
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'a-id', True, (('a', ['a', 'a']), 'a', 'a'), ((root_id, [root_id, root_id]), root_id, root_id), (('a', ['a', 'a']), 'a', 'a'), ((False, [False, False]), False, False), False)], entries)

    def test_not_in_base(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('add', ('foo', b'foo-id', 'file', b'a\nb\nc\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('add', ('bar', b'bar-id', 'file', b'd\ne\nf\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('add', ('bar', b'bar-id', 'file', b'd\ne\nf\n'))], revision_id=b'D-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('add', ('foo', b'foo-id', 'file', b'a\nb\nc\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'E-id', b'D-id'], [('modify', ('bar', b'd\ne\nf\nG\n'))], revision_id=b'G-id')
        builder.build_snapshot([b'D-id', b'E-id'], [], revision_id=b'F-id')
        merge_obj = self.make_merge_obj(builder, b'G-id')
        self.assertEqual([b'D-id', b'E-id'], [t.get_revision_id() for t in merge_obj._lca_trees])
        self.assertEqual(b'A-id', merge_obj.base_tree.get_revision_id())
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'bar-id', True, ((None, ['bar', 'bar']), 'bar', 'bar'), ((None, [root_id, root_id]), root_id, root_id), ((None, ['bar', 'bar']), 'bar', 'bar'), ((None, [False, False]), False, False), False)], entries)

    def test_not_in_this(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('a', b'a\nB\nb\nc\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('modify', ('a', b'a\nb\nC\nc\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('a', b'a\nB\nb\nC\nc\nE\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('unversion', 'a')], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        self.assertEqual([b'B-id', b'C-id'], [t.get_revision_id() for t in merge_obj._lca_trees])
        self.assertEqual(b'A-id', merge_obj.base_tree.get_revision_id())
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'a-id', True, (('a', ['a', 'a']), 'a', None), ((root_id, [root_id, root_id]), root_id, None), (('a', ['a', 'a']), 'a', None), ((False, [False, False]), False, None), False)], entries)

    def test_file_not_in_one_lca(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        self.assertEqual([b'B-id', b'C-id'], [t.get_revision_id() for t in merge_obj._lca_trees])
        self.assertEqual(b'A-id', merge_obj.base_tree.get_revision_id())
        entries = list(merge_obj._entries_lca())
        self.assertEqual([], entries)

    def test_not_in_other(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('unversion', 'a')], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'a-id', True, (('a', ['a', 'a']), None, 'a'), ((root_id, [root_id, root_id]), None, root_id), (('a', ['a', 'a']), None, 'a'), ((False, [False, False]), None, False), False)], entries)

    def test_not_in_other_or_lca(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('unversion', 'foo')], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        self.assertEqual([], entries)

    def test_not_in_other_mod_in_lca1_not_in_lca2(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'new-content\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('unversion', 'foo')], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'foo-id', True, (('foo', ['foo', None]), None, 'foo'), ((root_id, [root_id, None]), None, root_id), (('foo', ['foo', None]), None, 'foo'), ((False, [False, None]), None, False), False)], entries)

    def test_only_in_one_lca(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('unversion', 'a')], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        self.assertEqual([], entries)

    def test_only_in_other(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'a-id', True, ((None, [None, None]), 'a', None), ((None, [None, None]), root_id, None), ((None, [None, None]), 'a', None), ((None, [None, None]), False, None), False)], entries)

    def test_one_lca_supersedes(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('foo', b'E content\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'E-id', b'D-id'], [], revision_id=b'G-id')
        builder.build_snapshot([b'D-id', b'E-id'], [('modify', ('foo', b'F content\n'))], revision_id=b'F-id')
        merge_obj = self.make_merge_obj(builder, b'G-id')
        self.assertEqual([], list(merge_obj._entries_lca()))

    def test_one_lca_supersedes_path(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'A-id'], [('rename', ('foo', 'bar'))], revision_id=b'B-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('rename', ('foo', 'bing'))], revision_id=b'E-id')
        builder.build_snapshot([b'E-id', b'D-id'], [('rename', ('bing', 'barry'))], revision_id=b'G-id')
        builder.build_snapshot([b'D-id', b'E-id'], [('rename', ('bar', 'bing'))], revision_id=b'F-id')
        merge_obj = self.make_merge_obj(builder, b'G-id')
        self.expectFailure("We don't do an actual heads() check on lca values, or use the per-attribute graph", self.assertEqual, [], list(merge_obj._entries_lca()))

    def test_one_lca_accidentally_pruned(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'A-id'], [('rename', ('foo', 'bar'))], revision_id=b'B-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'E-id', b'D-id'], [('rename', ('foo', 'bar'))], revision_id=b'G-id')
        builder.build_snapshot([b'D-id', b'E-id'], [('rename', ('bar', 'bing'))], revision_id=b'F-id')
        merge_obj = self.make_merge_obj(builder, b'G-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.expectFailure('We prune values from BASE even when relevant.', self.assertEqual, [(b'foo-id', False, ((root_id, [root_id, root_id]), root_id, root_id), (('foo', ['bar', 'foo']), 'bar', 'bing'), ((False, [False, False]), False, False), False)], entries)

    def test_both_sides_revert(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'C content\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'foo-id', True, (('foo', ['foo', 'foo']), 'foo', 'foo'), ((root_id, [root_id, root_id]), root_id, root_id), (('foo', ['foo', 'foo']), 'foo', 'foo'), ((False, [False, False]), False, False), False)], entries)

    def test_different_lca_resolve_one_side_updates_content(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'C content\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        builder.build_snapshot([b'D-id'], [('modify', ('foo', b'F content\n'))], revision_id=b'F-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'foo-id', True, (('foo', ['foo', 'foo']), 'foo', 'foo'), ((root_id, [root_id, root_id]), root_id, root_id), (('foo', ['foo', 'foo']), 'foo', 'foo'), ((False, [False, False]), False, False), False)], entries)

    def test_same_lca_resolution_one_side_updates_content(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'A content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('modify', ('foo', b'C content\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('modify', ('foo', b'C content\n'))], revision_id=b'D-id')
        builder.build_snapshot([b'D-id'], [('modify', ('foo', b'F content\n'))], revision_id=b'F-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        self.expectFailure("We don't detect that LCA resolution was the same on both sides", self.assertEqual, [], entries)

    def test_only_path_changed(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('rename', ('a', 'b'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'a-id', False, (('a', ['a', 'a']), 'b', 'a'), ((root_id, [root_id, root_id]), root_id, root_id), (('a', ['a', 'a']), 'b', 'a'), ((False, [False, False]), False, False), False)], entries)

    def test_kind_changed(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('unversion', 'a'), ('flush', None), ('add', ('a', b'a-id', 'directory', None))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'a-id', True, (('a', ['a', 'a']), 'a', 'a'), ((root_id, [root_id, root_id]), root_id, root_id), (('a', ['a', 'a']), 'a', 'a'), ((False, [False, False]), False, False), False)], entries)

    def test_this_changed_kind(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('unversion', 'a'), ('flush', None), ('add', ('a', b'a-id', 'directory', None))], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id')
        entries = list(merge_obj._entries_lca())
        self.assertEqual([], entries)

    def test_interesting_file_in_this(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n')), ('add', ('b', b'b-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('a', b'new-content\n')), ('modify', ('b', b'new-content\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('rename', ('b', 'c'))], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id', interesting_files=['c'])
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'b-id', True, (('b', ['b', 'b']), 'b', 'c'), ((root_id, [root_id, root_id]), root_id, root_id), (('b', ['b', 'b']), 'b', 'c'), ((False, [False, False]), False, False), False)], entries)

    def test_interesting_file_in_base(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n')), ('add', ('c', b'c-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('rename', ('c', 'b'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('rename', ('c', 'b'))], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('a', b'new-content\n')), ('modify', ('b', b'new-content\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id', interesting_files=['c'])
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'c-id', True, (('c', ['b', 'b']), 'b', 'b'), ((root_id, [root_id, root_id]), root_id, root_id), (('c', ['b', 'b']), 'b', 'b'), ((False, [False, False]), False, False), False)], entries)

    def test_interesting_file_in_lca(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n')), ('add', ('b', b'b-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('rename', ('b', 'c'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('a', b'new-content\n')), ('modify', ('b', b'new-content\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('rename', ('c', 'b'))], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id', interesting_files=['c'])
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'b-id', True, (('b', ['c', 'b']), 'b', 'b'), ((root_id, [root_id, root_id]), root_id, root_id), (('b', ['c', 'b']), 'b', 'b'), ((False, [False, False]), False, False), False)], entries)

    def test_interesting_files(self):
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n')), ('add', ('b', b'b-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'C-id', b'B-id'], [('modify', ('a', b'new-content\n')), ('modify', ('b', b'new-content\n'))], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        merge_obj = self.make_merge_obj(builder, b'E-id', interesting_files=['b'])
        entries = list(merge_obj._entries_lca())
        root_id = b'a-root-id'
        self.assertEqual([(b'b-id', True, (('b', ['b', 'b']), 'b', 'b'), ((root_id, [root_id, root_id]), root_id, root_id), (('b', ['b', 'b']), 'b', 'b'), ((False, [False, False]), False, False), False)], entries)