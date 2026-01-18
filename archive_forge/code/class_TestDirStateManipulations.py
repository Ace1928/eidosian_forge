import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestDirStateManipulations(TestCaseWithDirState):

    def make_minimal_tree(self):
        tree1 = self.make_branch_and_memory_tree('tree1')
        tree1.lock_write()
        self.addCleanup(tree1.unlock)
        tree1.add('')
        revid1 = tree1.commit('foo')
        return (tree1, revid1)

    def test_update_minimal_updates_id_index(self):
        state = self.create_dirstate_with_root_and_subdir()
        self.addCleanup(state.unlock)
        id_index = state._get_id_index()
        self.assertEqual([b'a-root-value', b'subdir-id'], sorted(id_index))
        state.add('file-name', b'file-id', 'file', None, '')
        self.assertEqual([b'a-root-value', b'file-id', b'subdir-id'], sorted(id_index))
        state.update_minimal((b'', b'new-name', b'file-id'), b'f', path_utf8=b'new-name')
        self.assertEqual([b'a-root-value', b'file-id', b'subdir-id'], sorted(id_index))
        self.assertEqual([(b'', b'new-name', b'file-id')], sorted(id_index[b'file-id']))
        state._validate()

    def test_set_state_from_inventory_no_content_no_parents(self):
        tree1, revid1 = self.make_minimal_tree()
        inv = tree1.root_inventory
        root_id = inv.path2id('')
        expected_result = ([], [((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])])
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_state_from_inventory(inv)
            self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._header_state)
            self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
        except:
            state.unlock()
            raise
        else:
            self.check_state_with_reopen(expected_result, state)

    def test_set_state_from_scratch_no_parents(self):
        tree1, revid1 = self.make_minimal_tree()
        inv = tree1.root_inventory
        root_id = inv.path2id('')
        expected_result = ([], [((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])])
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_state_from_scratch(inv, [], [])
            self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._header_state)
            self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
        except:
            state.unlock()
            raise
        else:
            self.check_state_with_reopen(expected_result, state)

    def test_set_state_from_scratch_identical_parent(self):
        tree1, revid1 = self.make_minimal_tree()
        inv = tree1.root_inventory
        root_id = inv.path2id('')
        rev_tree1 = tree1.branch.repository.revision_tree(revid1)
        d_entry = (b'd', b'', 0, False, dirstate.DirState.NULLSTAT)
        parent_entry = (b'd', b'', 0, False, revid1)
        expected_result = ([revid1], [((b'', b'', root_id), [d_entry, parent_entry])])
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_state_from_scratch(inv, [(revid1, rev_tree1)], [])
            self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._header_state)
            self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
        except:
            state.unlock()
            raise
        else:
            self.check_state_with_reopen(expected_result, state)

    def test_set_state_from_inventory_preserves_hashcache(self):
        tree = self.make_branch_and_tree('.')
        with tree.lock_write():
            foo_contents = b'contents of foo'
            self.build_tree_contents([('foo', foo_contents)])
            tree.add('foo', ids=b'foo-id')
            foo_stat = os.stat('foo')
            foo_packed = dirstate.pack_stat(foo_stat)
            foo_sha = osutils.sha_string(foo_contents)
            foo_size = len(foo_contents)
            self.assertEqual(((b'', b'foo', b'foo-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT)]), tree._dirstate._get_entry(0, b'foo-id'))
            tree._dirstate.update_minimal((b'', b'foo', b'foo-id'), b'f', False, foo_sha, foo_packed, foo_size, b'foo')
            self.assertEqual(((b'', b'foo', b'foo-id'), [(b'f', foo_sha, foo_size, False, foo_packed)]), tree._dirstate._get_entry(0, b'foo-id'))
            inv = tree._get_root_inventory()
            self.assertTrue(inv.has_id(b'foo-id'))
            self.assertTrue(inv.has_filename('foo'))
            inv.add_path('bar', 'file', b'bar-id')
            tree._dirstate._validate()
            tree._dirstate.set_state_from_inventory(inv)
            tree._dirstate._validate()
        with tree.lock_read():
            state = tree._dirstate
            state._validate()
            foo_tuple = state._get_entry(0, path_utf8=b'foo')
            self.assertEqual(((b'', b'foo', b'foo-id'), [(b'f', foo_sha, len(foo_contents), False, dirstate.pack_stat(foo_stat))]), foo_tuple)

    def test_set_state_from_inventory_mixed_paths(self):
        tree1 = self.make_branch_and_tree('tree1')
        self.build_tree(['tree1/a/', 'tree1/a/b/', 'tree1/a-b/', 'tree1/a/b/foo', 'tree1/a-b/bar'])
        tree1.lock_write()
        try:
            tree1.add(['a', 'a/b', 'a-b', 'a/b/foo', 'a-b/bar'], ids=[b'a-id', b'b-id', b'a-b-id', b'foo-id', b'bar-id'])
            tree1.commit('rev1', rev_id=b'rev1')
            root_id = tree1.path2id('')
            inv = tree1.root_inventory
        finally:
            tree1.unlock()
        expected_result1 = [(b'', b'', root_id, b'd'), (b'', b'a', b'a-id', b'd'), (b'', b'a-b', b'a-b-id', b'd'), (b'a', b'b', b'b-id', b'd'), (b'a/b', b'foo', b'foo-id', b'f'), (b'a-b', b'bar', b'bar-id', b'f')]
        expected_result2 = [(b'', b'', root_id, b'd'), (b'', b'a', b'a-id', b'd'), (b'', b'a-b', b'a-b-id', b'd'), (b'a-b', b'bar', b'bar-id', b'f')]
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_state_from_inventory(inv)
            values = []
            for entry in state._iter_entries():
                values.append(entry[0] + entry[1][0][:1])
            self.assertEqual(expected_result1, values)
            inv.delete(b'b-id')
            state.set_state_from_inventory(inv)
            values = []
            for entry in state._iter_entries():
                values.append(entry[0] + entry[1][0][:1])
            self.assertEqual(expected_result2, values)
        finally:
            state.unlock()

    def test_set_path_id_no_parents(self):
        """The id of a path can be changed trivally with no parents."""
        state = dirstate.DirState.initialize('dirstate')
        try:
            root_entry = ((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, b'x' * 32)])
            self.assertEqual([root_entry], list(state._iter_entries()))
            self.assertEqual(root_entry, state._get_entry(0, path_utf8=b''))
            self.assertEqual(root_entry, state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
            self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'second-root-id'))
            state.set_path_id(b'', b'second-root-id')
            new_root_entry = ((b'', b'', b'second-root-id'), [(b'd', b'', 0, False, b'x' * 32)])
            expected_rows = [new_root_entry]
            self.assertEqual(expected_rows, list(state._iter_entries()))
            self.assertEqual(new_root_entry, state._get_entry(0, path_utf8=b''))
            self.assertEqual(new_root_entry, state._get_entry(0, fileid_utf8=b'second-root-id'))
            self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            state._validate()
            self.assertEqual(expected_rows, list(state._iter_entries()))
        finally:
            state.unlock()

    def test_set_path_id_with_parents(self):
        """Set the root file id in a dirstate with parents"""
        mt = self.make_branch_and_tree('mt')
        mt.set_root_id(b'TREE_ROOT')
        mt.commit('foo', rev_id=b'parent-revid')
        rt = mt.branch.repository.revision_tree(b'parent-revid')
        state = dirstate.DirState.initialize('dirstate')
        state._validate()
        try:
            state.set_parent_trees([(b'parent-revid', rt)], ghosts=[])
            root_entry = ((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, b'x' * 32), (b'd', b'', 0, False, b'parent-revid')])
            self.assertEqual(root_entry, state._get_entry(0, path_utf8=b''))
            self.assertEqual(root_entry, state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
            self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'Asecond-root-id'))
            state.set_path_id(b'', b'Asecond-root-id')
            state._validate()
            old_root_entry = ((b'', b'', b'TREE_ROOT'), [(b'a', b'', 0, False, b''), (b'd', b'', 0, False, b'parent-revid')])
            new_root_entry = ((b'', b'', b'Asecond-root-id'), [(b'd', b'', 0, False, b''), (b'a', b'', 0, False, b'')])
            expected_rows = [new_root_entry, old_root_entry]
            state._validate()
            self.assertEqual(expected_rows, list(state._iter_entries()))
            self.assertEqual(new_root_entry, state._get_entry(0, path_utf8=b''))
            self.assertEqual(old_root_entry, state._get_entry(1, path_utf8=b''))
            self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
            self.assertEqual(old_root_entry, state._get_entry(1, fileid_utf8=b'TREE_ROOT'))
            self.assertEqual(new_root_entry, state._get_entry(0, fileid_utf8=b'Asecond-root-id'))
            self.assertEqual((None, None), state._get_entry(1, fileid_utf8=b'Asecond-root-id'))
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            state._validate()
            self.assertEqual(expected_rows, list(state._iter_entries()))
        finally:
            state.unlock()
        state.lock_write()
        try:
            state._validate()
            state.set_path_id(b'', b'tree-root-2')
            state._validate()
        finally:
            state.unlock()

    def test_set_parent_trees_no_content(self):
        tree1 = self.make_branch_and_memory_tree('tree1')
        tree1.lock_write()
        try:
            tree1.add('')
            revid1 = tree1.commit('foo')
        finally:
            tree1.unlock()
        branch2 = tree1.branch.controldir.clone('tree2').open_branch()
        tree2 = memorytree.MemoryTree.create_on_branch(branch2)
        tree2.lock_write()
        try:
            revid2 = tree2.commit('foo')
            root_id = tree2.path2id('')
        finally:
            tree2.unlock()
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_path_id(b'', root_id)
            state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (revid2, tree2.branch.repository.revision_tree(revid2)), (b'ghost-rev', None)), [b'ghost-rev'])
            state._validate()
            state.save()
            state._validate()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_write()
        try:
            self.assertEqual([revid1, revid2, b'ghost-rev'], state.get_parent_ids())
            list(state._iter_entries())
            state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (b'ghost-rev', None)), [b'ghost-rev'])
            state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (revid2, tree2.branch.repository.revision_tree(revid2)), (b'ghost-rev', tree2.branch.repository.revision_tree(_mod_revision.NULL_REVISION))), [b'ghost-rev'])
            self.assertEqual([revid1, revid2, b'ghost-rev'], state.get_parent_ids())
            self.assertEqual([b'ghost-rev'], state.get_ghosts())
            self.assertEqual([((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, revid1), (b'd', b'', 0, False, revid1)])], list(state._iter_entries()))
        finally:
            state.unlock()

    def test_set_parent_trees_file_missing_from_tree(self):
        tree1 = self.make_branch_and_memory_tree('tree1')
        tree1.lock_write()
        try:
            tree1.add('')
            tree1.add(['a file'], ['file'], [b'file-id'])
            tree1.put_file_bytes_non_atomic('a file', b'file-content')
            revid1 = tree1.commit('foo')
        finally:
            tree1.unlock()
        branch2 = tree1.branch.controldir.clone('tree2').open_branch()
        tree2 = memorytree.MemoryTree.create_on_branch(branch2)
        tree2.lock_write()
        try:
            tree2.put_file_bytes_non_atomic('a file', b'new file-content')
            revid2 = tree2.commit('foo')
            root_id = tree2.path2id('')
        finally:
            tree2.unlock()
        expected_result = ([revid1, revid2], [((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, revid1), (b'd', b'', 0, False, revid1)]), ((b'', b'a file', b'file-id'), [(b'a', b'', 0, False, b''), (b'f', b'2439573625385400f2a669657a7db6ae7515d371', 12, False, revid1), (b'f', b'542e57dc1cda4af37cb8e55ec07ce60364bb3c7d', 16, False, revid2)])])
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_path_id(b'', root_id)
            state.set_parent_trees(((revid1, tree1.branch.repository.revision_tree(revid1)), (revid2, tree2.branch.repository.revision_tree(revid2))), [])
        except:
            state.unlock()
            raise
        else:
            self.check_state_with_reopen(expected_result, state)

    def test_add_path_to_root_no_parents_all_data(self):
        self.build_tree(['a file'])
        stat = os.lstat('a file')
        state = dirstate.DirState.initialize('dirstate')
        expected_entries = [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)]), ((b'', b'a file', b'a-file-id'), [(b'f', b'1' * 20, 19, False, dirstate.pack_stat(stat))])]
        try:
            state.add('a file', b'a-file-id', 'file', stat, b'1' * 20)
            self.assertEqual(expected_entries, list(state._iter_entries()))
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        self.addCleanup(state.unlock)
        self.assertEqual(expected_entries, list(state._iter_entries()))

    def test_add_path_to_unversioned_directory(self):
        """Adding a path to an unversioned directory should error.

        This is a duplicate of TestWorkingTree.test_add_in_unversioned,
        once dirstate is stable and if it is merged with WorkingTree3, consider
        removing this copy of the test.
        """
        self.build_tree(['unversioned/', 'unversioned/a file'])
        state = dirstate.DirState.initialize('dirstate')
        self.addCleanup(state.unlock)
        self.assertRaises(errors.NotVersionedError, state.add, 'unversioned/a file', b'a-file-id', 'file', None, None)

    def test_add_directory_to_root_no_parents_all_data(self):
        self.build_tree(['a dir/'])
        stat = os.lstat('a dir')
        expected_entries = [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)]), ((b'', b'a dir', b'a dir id'), [(b'd', b'', 0, False, dirstate.pack_stat(stat))])]
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.add('a dir', b'a dir id', 'directory', stat, None)
            self.assertEqual(expected_entries, list(state._iter_entries()))
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        self.addCleanup(state.unlock)
        state._validate()
        self.assertEqual(expected_entries, list(state._iter_entries()))

    def _test_add_symlink_to_root_no_parents_all_data(self, link_name, target):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        os.symlink(target, link_name)
        stat = os.lstat(link_name)
        expected_entries = [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)]), ((b'', link_name.encode('UTF-8'), b'a link id'), [(b'l', target.encode('UTF-8'), stat[6], False, dirstate.pack_stat(stat))])]
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.add(link_name, b'a link id', 'symlink', stat, target.encode('UTF-8'))
            self.assertEqual(expected_entries, list(state._iter_entries()))
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        self.addCleanup(state.unlock)
        self.assertEqual(expected_entries, list(state._iter_entries()))

    def test_add_symlink_to_root_no_parents_all_data(self):
        self._test_add_symlink_to_root_no_parents_all_data('a link', 'target')

    def test_add_symlink_unicode_to_root_no_parents_all_data(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_add_symlink_to_root_no_parents_all_data('€link', 'targ€et')

    def test_add_directory_and_child_no_parents_all_data(self):
        self.build_tree(['a dir/', 'a dir/a file'])
        dirstat = os.lstat('a dir')
        filestat = os.lstat('a dir/a file')
        expected_entries = [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)]), ((b'', b'a dir', b'a dir id'), [(b'd', b'', 0, False, dirstate.pack_stat(dirstat))]), ((b'a dir', b'a file', b'a-file-id'), [(b'f', b'1' * 20, 25, False, dirstate.pack_stat(filestat))])]
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.add('a dir', b'a dir id', 'directory', dirstat, None)
            state.add('a dir/a file', b'a-file-id', 'file', filestat, b'1' * 20)
            self.assertEqual(expected_entries, list(state._iter_entries()))
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        self.addCleanup(state.unlock)
        self.assertEqual(expected_entries, list(state._iter_entries()))

    def test_add_tree_reference(self):
        state = dirstate.DirState.initialize('dirstate')
        expected_entry = ((b'', b'subdir', b'subdir-id'), [(b't', b'subtree-123123', 0, False, b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')])
        try:
            state.add('subdir', b'subdir-id', 'tree-reference', None, b'subtree-123123')
            entry = state._get_entry(0, b'subdir-id', b'subdir')
            self.assertEqual(entry, expected_entry)
            state._validate()
            state.save()
        finally:
            state.unlock()
        state.lock_read()
        self.addCleanup(state.unlock)
        state._validate()
        entry2 = state._get_entry(0, b'subdir-id', b'subdir')
        self.assertEqual(entry, entry2)
        self.assertEqual(entry, expected_entry)
        entry2 = state._get_entry(0, fileid_utf8=b'subdir-id')
        self.assertEqual(entry, expected_entry)

    def test_add_forbidden_names(self):
        state = dirstate.DirState.initialize('dirstate')
        self.addCleanup(state.unlock)
        self.assertRaises(errors.BzrError, state.add, '.', b'ass-id', 'directory', None, None)
        self.assertRaises(errors.BzrError, state.add, '..', b'ass-id', 'directory', None, None)

    def test_set_state_with_rename_b_a_bug_395556(self):
        tree1 = self.make_branch_and_tree('tree1')
        self.build_tree(['tree1/b'])
        with tree1.lock_write():
            tree1.add(['b'], ids=[b'b-id'])
            root_id = tree1.path2id('')
            inv = tree1.root_inventory
            state = dirstate.DirState.initialize('dirstate')
            try:
                state.set_state_from_inventory(inv)
                inv.rename(b'b-id', root_id, 'a')
                state.set_state_from_inventory(inv)
                expected_result1 = [(b'', b'', root_id, b'd'), (b'', b'a', b'b-id', b'f')]
                values = []
                for entry in state._iter_entries():
                    values.append(entry[0] + entry[1][0][:1])
                self.assertEqual(expected_result1, values)
            finally:
                state.unlock()