import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
class TestHookMergeFileContent(TestCaseWithTransport):
    """Tests that the 'merge_file_content' hook is invoked."""

    def setUp(self):
        super().setUp()
        self.hook_log = []

    def install_hook_inactive(self):

        def inactive_factory(merger):
            self.hook_log.append(('inactive',))
            return None
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', inactive_factory, 'test hook (inactive)')

    def install_hook_noop(self):
        test = self

        class HookNA(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                test.hook_log.append(('no-op',))
                return ('not_applicable', None)

        def hook_na_factory(merger):
            return HookNA(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_na_factory, 'test hook (no-op)')

    def install_hook_success(self):
        test = self

        class HookSuccess(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                test.hook_log.append(('success',))
                if merge_params.this_path == 'name1':
                    return ('success', [b'text-merged-by-hook'])
                return ('not_applicable', None)

        def hook_success_factory(merger):
            return HookSuccess(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_success_factory, 'test hook (success)')

    def install_hook_conflict(self):
        test = self

        class HookConflict(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                test.hook_log.append(('conflict',))
                if merge_params.this_path == 'name1':
                    return ('conflicted', [b'text-with-conflict-markers-from-hook'])
                return ('not_applicable', None)

        def hook_conflict_factory(merger):
            return HookConflict(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_conflict_factory, 'test hook (delete)')

    def install_hook_delete(self):
        test = self

        class HookDelete(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                test.hook_log.append(('delete',))
                if merge_params.this_path == 'name1':
                    return ('delete', None)
                return ('not_applicable', None)

        def hook_delete_factory(merger):
            return HookDelete(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_delete_factory, 'test hook (delete)')

    def install_hook_log_lines(self):
        """Install a hook that saves the get_lines for the this, base and other
        versions of the file.
        """
        test = self

        class HookLogLines(_mod_merge.AbstractPerFileMerger):

            def merge_contents(self, merge_params):
                test.hook_log.append(('log_lines', merge_params.this_lines, merge_params.other_lines, merge_params.base_lines))
                return ('not_applicable', None)

        def hook_log_lines_factory(merger):
            return HookLogLines(merger)
        _mod_merge.Merger.hooks.install_named_hook('merge_file_content', hook_log_lines_factory, 'test hook (log_lines)')

    def make_merge_builder(self):
        builder = MergeBuilder(self.test_base_dir)
        self.addCleanup(builder.cleanup)
        return builder

    def create_file_needing_contents_merge(self, builder, name):
        file_id = name.encode('ascii') + b'-id'
        transid = builder.add_file(builder.root(), name, b'text1', True, file_id=file_id)
        builder.change_contents(transid, other=b'text4', this=b'text3')

    def test_change_vs_change(self):
        """Hook is used for (changed, changed)"""
        self.install_hook_success()
        builder = self.make_merge_builder()
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_contents(name1, other=b'text4', this=b'text3')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        with builder.this.get_file('name1') as f:
            self.assertEqual(f.read(), b'text-merged-by-hook')

    def test_change_vs_deleted(self):
        """Hook is used for (changed, deleted)"""
        self.install_hook_success()
        builder = self.make_merge_builder()
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_contents(name1, this=b'text2')
        builder.remove_file(name1, other=True)
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        with builder.this.get_file('name1') as f:
            self.assertEqual(f.read(), b'text-merged-by-hook')

    def test_result_can_be_delete(self):
        """A hook's result can be the deletion of a file."""
        self.install_hook_delete()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        self.assertFalse(builder.this.is_versioned('name1'))
        self.assertEqual([], list(builder.this.list_files()))

    def test_result_can_be_conflict(self):
        """A hook's result can be a conflict."""
        self.install_hook_conflict()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(1, len(conflicts))
        [conflict] = conflicts
        self.assertEqual('text conflict', conflict.typestring)
        if builder.this.supports_file_ids:
            self.assertEqual(conflict.file_id, builder.this.path2id('name1'))
        with builder.this.get_file('name1') as f:
            self.assertEqual(f.read(), b'text-with-conflict-markers-from-hook')

    def test_can_access_this_other_and_base_versions(self):
        """The hook function can call params.merger.get_lines to access the
        THIS/OTHER/BASE versions of the file.
        """
        self.install_hook_log_lines()
        builder = self.make_merge_builder()
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_contents(name1, this=b'text2', other=b'text3')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('log_lines', [b'text2'], [b'text3'], [b'text1'])], self.hook_log)

    def test_chain_when_not_active(self):
        """When a hook function returns None, merging still works."""
        self.install_hook_inactive()
        self.install_hook_success()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        with builder.this.get_file('name1') as f:
            self.assertEqual(f.read(), b'text-merged-by-hook')
        self.assertEqual([('inactive',), ('success',)], self.hook_log)

    def test_chain_when_not_applicable(self):
        """When a hook function returns not_applicable, the next function is
        tried (when one exists).
        """
        self.install_hook_noop()
        self.install_hook_success()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual(conflicts, [])
        with builder.this.get_file('name1') as f:
            self.assertEqual(f.read(), b'text-merged-by-hook')
        self.assertEqual([('no-op',), ('success',)], self.hook_log)

    def test_chain_stops_after_success(self):
        """When a hook function returns success, no later functions are tried.
        """
        self.install_hook_success()
        self.install_hook_noop()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('success',)], self.hook_log)

    def test_chain_stops_after_conflict(self):
        """When a hook function returns conflict, no later functions are tried.
        """
        self.install_hook_conflict()
        self.install_hook_noop()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('conflict',)], self.hook_log)

    def test_chain_stops_after_delete(self):
        """When a hook function returns delete, no later functions are tried.
        """
        self.install_hook_delete()
        self.install_hook_noop()
        builder = self.make_merge_builder()
        self.create_file_needing_contents_merge(builder, 'name1')
        conflicts = builder.merge(self.merge_type)
        self.assertEqual([('delete',)], self.hook_log)