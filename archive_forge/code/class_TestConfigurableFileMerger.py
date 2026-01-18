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
class TestConfigurableFileMerger(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.calls = []

    def get_merger_factory(self):
        calls = self.calls

        class FooMerger(_mod_merge.ConfigurableFileMerger):
            name_prefix = 'foo'
            default_files = ['bar']

            def merge_text(self, params):
                calls.append('merge_text')
                return ('not_applicable', None)

        def factory(merger):
            result = FooMerger(merger)
            self.assertEqual(None, result.affected_files)
            self.merger = result
            return result
        return factory

    def _install_hook(self, factory):
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', factory, 'test factory')

    def make_builder(self):
        builder = test_merge_core.MergeBuilder(self.test_base_dir)
        self.addCleanup(builder.cleanup)
        return builder

    def make_text_conflict(self, file_name='bar'):
        factory = self.get_merger_factory()
        self._install_hook(factory)
        builder = self.make_builder()
        trans_ids = builder.add_file(builder.root(), file_name, b'text1', True, file_id=b'bar-id')
        builder.change_contents(trans_ids, other=b'text4', this=b'text3')
        return builder

    def make_kind_change(self):
        factory = self.get_merger_factory()
        self._install_hook(factory)
        builder = self.make_builder()
        builder.add_file(builder.root(), 'bar', b'text1', True, this=False, file_id=b'bar-id')
        builder.add_dir(builder.root(), 'bar-id', base=False, other=False, file_id=b'bar-dir')
        return builder

    def test_uses_this_branch(self):
        builder = self.make_text_conflict()
        with builder.make_preview_transform() as tt:
            pass

    def test_affected_files_cached(self):
        """Ensures that the config variable is cached"""
        builder = self.make_text_conflict()
        conflicts = builder.merge()
        self.assertEqual(['bar'], self.merger.affected_files)
        self.assertEqual(1, len(conflicts))

    def test_hook_called_for_text_conflicts(self):
        builder = self.make_text_conflict()
        builder.merge()
        self.assertEqual(['merge_text'], self.calls)

    def test_hook_not_called_for_kind_change(self):
        builder = self.make_kind_change()
        builder.merge()
        self.assertEqual([], self.calls)

    def test_hook_not_called_for_other_files(self):
        builder = self.make_text_conflict('foobar')
        builder.merge()
        self.assertEqual([], self.calls)