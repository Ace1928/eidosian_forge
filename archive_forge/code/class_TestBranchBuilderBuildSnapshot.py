from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
class TestBranchBuilderBuildSnapshot(tests.TestCaseWithMemoryTransport):

    def assertTreeShape(self, expected_shape, tree):
        """Check that the tree shape matches expectations."""
        tree.lock_read()
        try:
            entries = [(path, ie.file_id, ie.kind) for path, ie in tree.iter_entries_by_dir()]
        finally:
            tree.unlock()
        self.assertEqual(expected_shape, entries)

    def build_a_rev(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id1 = builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'contents'))], revision_id=b'A-id')
        self.assertEqual(b'A-id', rev_id1)
        return builder

    def test_add_one_file(self):
        builder = self.build_a_rev()
        branch = builder.get_branch()
        self.assertEqual((1, b'A-id'), branch.last_revision_info())
        rev_tree = branch.repository.revision_tree(b'A-id')
        rev_tree.lock_read()
        self.addCleanup(rev_tree.unlock)
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], rev_tree)
        self.assertEqual(b'contents', rev_tree.get_file_text('a'))

    def test_add_second_file(self):
        builder = self.build_a_rev()
        rev_id2 = builder.build_snapshot(None, [('add', ('b', b'b-id', 'file', b'content_b'))], revision_id=b'B-id')
        self.assertEqual(b'B-id', rev_id2)
        branch = builder.get_branch()
        self.assertEqual((2, rev_id2), branch.last_revision_info())
        rev_tree = branch.repository.revision_tree(rev_id2)
        rev_tree.lock_read()
        self.addCleanup(rev_tree.unlock)
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'file')], rev_tree)
        self.assertEqual(b'content_b', rev_tree.get_file_text('b'))

    def test_add_empty_dir(self):
        builder = self.build_a_rev()
        rev_id2 = builder.build_snapshot(None, [('add', ('b', b'b-id', 'directory', None))], revision_id=b'B-id')
        rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'directory')], rev_tree)

    def test_commit_timestamp(self):
        builder = self.make_branch_builder('foo')
        rev_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None))], timestamp=1234567890)
        rev = builder.get_branch().repository.get_revision(rev_id)
        self.assertEqual(1234567890, int(rev.timestamp))

    def test_commit_message_default(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None))])
        branch = builder.get_branch()
        rev = branch.repository.get_revision(rev_id)
        self.assertEqual('commit 1', rev.message)

    def test_commit_message_supplied(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None))], message='Foo')
        branch = builder.get_branch()
        rev = branch.repository.get_revision(rev_id)
        self.assertEqual('Foo', rev.message)

    def test_commit_message_callback(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None))], message_callback=lambda x: 'Foo')
        branch = builder.get_branch()
        rev = branch.repository.get_revision(rev_id)
        self.assertEqual('Foo', rev.message)

    def test_modify_file(self):
        builder = self.build_a_rev()
        rev_id2 = builder.build_snapshot(None, [('modify', ('a', b'new\ncontent\n'))], revision_id=b'B-id')
        self.assertEqual(b'B-id', rev_id2)
        branch = builder.get_branch()
        rev_tree = branch.repository.revision_tree(rev_id2)
        rev_tree.lock_read()
        self.addCleanup(rev_tree.unlock)
        self.assertEqual(b'new\ncontent\n', rev_tree.get_file_text('a'))

    def test_delete_file(self):
        builder = self.build_a_rev()
        rev_id2 = builder.build_snapshot(None, [('unversion', 'a')], revision_id=b'B-id')
        self.assertEqual(b'B-id', rev_id2)
        branch = builder.get_branch()
        rev_tree = branch.repository.revision_tree(rev_id2)
        rev_tree.lock_read()
        self.addCleanup(rev_tree.unlock)
        self.assertTreeShape([('', b'a-root-id', 'directory')], rev_tree)

    def test_delete_directory(self):
        builder = self.build_a_rev()
        rev_id2 = builder.build_snapshot(None, [('add', ('b', b'b-id', 'directory', None)), ('add', ('b/c', b'c-id', 'file', b'foo\n')), ('add', ('b/d', b'd-id', 'directory', None)), ('add', ('b/d/e', b'e-id', 'file', b'eff\n'))], revision_id=b'B-id')
        rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'directory'), ('b/c', b'c-id', 'file'), ('b/d', b'd-id', 'directory'), ('b/d/e', b'e-id', 'file')], rev_tree)
        builder.build_snapshot(None, [('unversion', 'b')], revision_id=b'C-id')
        rev_tree = builder.get_branch().repository.revision_tree(b'C-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], rev_tree)

    def test_unknown_action(self):
        builder = self.build_a_rev()
        e = self.assertRaises(ValueError, builder.build_snapshot, None, [('weirdo', ('foo',))], revision_id=b'B-id')
        self.assertEqual('Unknown build action: "weirdo"', str(e))

    def test_rename(self):
        builder = self.build_a_rev()
        builder.build_snapshot(None, [('rename', ('a', 'b'))], revision_id=b'B-id')
        rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('b', b'a-id', 'file')], rev_tree)

    def test_rename_into_subdir(self):
        builder = self.build_a_rev()
        builder.build_snapshot(None, [('add', ('dir', b'dir-id', 'directory', None)), ('rename', ('a', 'dir/a'))], revision_id=b'B-id')
        rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('dir', b'dir-id', 'directory'), ('dir/a', b'a-id', 'file')], rev_tree)

    def test_rename_out_of_unversioned_subdir(self):
        builder = self.build_a_rev()
        builder.build_snapshot(None, [('add', ('dir', b'dir-id', 'directory', None)), ('rename', ('a', 'dir/a'))], revision_id=b'B-id')
        builder.build_snapshot(None, [('rename', ('dir/a', 'a')), ('unversion', 'dir')], revision_id=b'C-id')
        rev_tree = builder.get_branch().repository.revision_tree(b'C-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], rev_tree)

    def test_set_parent(self):
        builder = self.build_a_rev()
        builder.start_series()
        self.addCleanup(builder.finish_series)
        builder.build_snapshot([b'A-id'], [('modify', ('a', b'new\ncontent\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('add', ('c', b'c-id', 'file', b'alt\ncontent\n'))], revision_id=b'C-id')
        repo = builder.get_branch().repository
        self.assertEqual({b'B-id': (b'A-id',), b'C-id': (b'A-id',)}, repo.get_parent_map([b'B-id', b'C-id']))
        b_tree = repo.revision_tree(b'B-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], b_tree)
        self.assertEqual(b'new\ncontent\n', b_tree.get_file_text('a'))
        c_tree = repo.revision_tree(b'C-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('c', b'c-id', 'file')], c_tree)
        self.assertEqual(b'contents', c_tree.get_file_text('a'))
        self.assertEqual(b'alt\ncontent\n', c_tree.get_file_text('c'))

    def test_set_merge_parent(self):
        builder = self.build_a_rev()
        builder.start_series()
        self.addCleanup(builder.finish_series)
        builder.build_snapshot([b'A-id'], [('add', ('b', b'b-id', 'file', b'b\ncontent\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('add', ('c', b'c-id', 'file', b'alt\ncontent\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        repo = builder.get_branch().repository
        self.assertEqual({b'B-id': (b'A-id',), b'C-id': (b'A-id',), b'D-id': (b'B-id', b'C-id')}, repo.get_parent_map([b'B-id', b'C-id', b'D-id']))
        d_tree = repo.revision_tree(b'D-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'file')], d_tree)

    def test_set_merge_parent_and_contents(self):
        builder = self.build_a_rev()
        builder.start_series()
        self.addCleanup(builder.finish_series)
        builder.build_snapshot([b'A-id'], [('add', ('b', b'b-id', 'file', b'b\ncontent\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [('add', ('c', b'c-id', 'file', b'alt\ncontent\n'))], revision_id=b'C-id')
        builder.build_snapshot([b'B-id', b'C-id'], [('add', ('c', b'c-id', 'file', b'alt\ncontent\n'))], revision_id=b'D-id')
        repo = builder.get_branch().repository
        self.assertEqual({b'B-id': (b'A-id',), b'C-id': (b'A-id',), b'D-id': (b'B-id', b'C-id')}, repo.get_parent_map([b'B-id', b'C-id', b'D-id']))
        d_tree = repo.revision_tree(b'D-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'file'), ('c', b'c-id', 'file')], d_tree)
        self.assertEqual(b'C-id', d_tree.get_file_revision('c'))

    def test_set_parent_to_null(self):
        builder = self.build_a_rev()
        builder.start_series()
        self.addCleanup(builder.finish_series)
        builder.build_snapshot([], [('add', ('', None, 'directory', None))], revision_id=b'B-id')
        repo = builder.get_branch().repository
        self.assertEqual({b'A-id': (_mod_revision.NULL_REVISION,), b'B-id': (_mod_revision.NULL_REVISION,)}, repo.get_parent_map([b'A-id', b'B-id']))

    def test_start_finish_series(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        builder.start_series()
        try:
            self.assertIsNot(None, builder._tree)
            self.assertEqual('w', builder._tree._lock_mode)
            self.assertTrue(builder._branch.is_locked())
        finally:
            builder.finish_series()
        self.assertIs(None, builder._tree)
        self.assertFalse(builder._branch.is_locked())

    def test_ghost_mainline_history(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        builder.start_series()
        try:
            builder.build_snapshot([b'ghost'], [('add', ('', b'ROOT_ID', 'directory', ''))], allow_leftmost_as_ghost=True, revision_id=b'tip')
        finally:
            builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        self.assertEqual((b'ghost',), b.repository.get_graph().get_parent_map([b'tip'])[b'tip'])

    def test_unversion_root_add_new_root(self):
        builder = BranchBuilder(self.get_transport().clone('foo'))
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', ''))], revision_id=b'rev-1')
        builder.build_snapshot(None, [('unversion', ''), ('add', ('', b'my-root', 'directory', ''))], revision_id=b'rev-2')
        builder.finish_series()
        rev_tree = builder.get_branch().repository.revision_tree(b'rev-2')
        self.assertTreeShape([('', b'my-root', 'directory')], rev_tree)

    def test_empty_flush(self):
        """A flush with no actions before it is a no-op."""
        builder = BranchBuilder(self.get_transport().clone('foo'))
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', ''))], revision_id=b'rev-1')
        builder.build_snapshot(None, [('flush', None)], revision_id=b'rev-2')
        builder.finish_series()
        rev_tree = builder.get_branch().repository.revision_tree(b'rev-2')
        self.assertTreeShape([('', b'TREE_ROOT', 'directory')], rev_tree)

    def test_kind_change(self):
        """It's possible to change the kind of an entry in a single snapshot
        with a bit of help from the 'flush' action.
        """
        builder = BranchBuilder(self.get_transport().clone('foo'))
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot(None, [('unversion', 'a'), ('flush', None), ('add', ('a', b'a-id', 'directory', None))], revision_id=b'B-id')
        builder.finish_series()
        rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
        self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'directory')], rev_tree)

    def test_pivot_root(self):
        """It's possible (albeit awkward) to move an existing dir to the root
        in a single snapshot by using unversion then flush then add.
        """
        builder = BranchBuilder(self.get_transport().clone('foo'))
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'orig-root', 'directory', None)), ('add', ('dir', b'dir-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot(None, [('unversion', ''), ('flush', None), ('add', ('', b'dir-id', 'directory', None))], revision_id=b'B-id')
        builder.finish_series()
        rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
        self.assertTreeShape([('', b'dir-id', 'directory')], rev_tree)