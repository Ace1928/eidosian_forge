import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestUpdateEntry(test_dirstate.TestCaseWithDirState):
    """Test the DirState.update_entry functions"""
    scenarios = multiply_scenarios(dir_reader_scenarios(), ue_scenarios)
    update_entry = None

    def setUp(self):
        super().setUp()
        self.overrideAttr(dirstate, 'update_entry', self.update_entry)

    def get_state_with_a(self):
        """Create a DirState tracking a single object named 'a'"""
        state = test_dirstate.InstrumentedDirState.initialize('dirstate')
        self.addCleanup(state.unlock)
        state.add('a', b'a-id', 'file', None, b'')
        entry = state._get_entry(0, path_utf8=b'a')
        return (state, entry)

    def test_observed_sha1_cachable(self):
        state, entry = self.get_state_with_a()
        state.save()
        atime = time.time() - 10
        self.build_tree(['a'])
        statvalue = test_dirstate._FakeStat.from_stat(os.lstat('a'))
        statvalue.st_mtime = statvalue.st_ctime = atime
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        state._observed_sha1(entry, b'foo', statvalue)
        self.assertEqual(b'foo', entry[1][0][1])
        packed_stat = dirstate.pack_stat(statvalue)
        self.assertEqual(packed_stat, entry[1][0][4])
        self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)

    def test_observed_sha1_not_cachable(self):
        state, entry = self.get_state_with_a()
        state.save()
        oldval = entry[1][0][1]
        oldstat = entry[1][0][4]
        self.build_tree(['a'])
        statvalue = os.lstat('a')
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        state._observed_sha1(entry, 'foo', statvalue)
        self.assertEqual(oldval, entry[1][0][1])
        self.assertEqual(oldstat, entry[1][0][4])
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)

    def test_update_entry(self):
        state, _ = self.get_state_with_a()
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        empty_revid = tree.commit('empty')
        self.build_tree(['tree/a'])
        tree.add(['a'], ids=[b'a-id'])
        with_a_id = tree.commit('with_a')
        self.addCleanup(tree.unlock)
        state.set_parent_trees([(empty_revid, tree.branch.repository.revision_tree(empty_revid))], [])
        entry = state._get_entry(0, path_utf8=b'a')
        self.build_tree(['a'])
        self.assertEqual((b'', b'a', b'a-id'), entry[0])
        self.assertEqual((b'f', b'', 0, False, dirstate.DirState.NULLSTAT), entry[1][0])
        state.save()
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        stat_value = os.lstat('a')
        packed_stat = dirstate.pack_stat(stat_value)
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(None, link_or_sha1)
        self.assertEqual((b'f', b'', 14, False, dirstate.DirState.NULLSTAT), entry[1][0])
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        mode = stat_value.st_mode
        self.assertEqual([('is_exec', mode, False)], state._log)
        state.save()
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        state.adjust_time(-10)
        del state._log[:]
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual([('is_exec', mode, False)], state._log)
        self.assertEqual(None, link_or_sha1)
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        self.assertEqual((b'f', b'', 14, False, dirstate.DirState.NULLSTAT), entry[1][0])
        state.save()
        state.adjust_time(+20)
        del state._log[:]
        link_or_sha1 = dirstate.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(None, link_or_sha1)
        self.assertEqual([('is_exec', mode, False)], state._log)
        self.assertEqual((b'f', b'', 14, False, dirstate.DirState.NULLSTAT), entry[1][0])
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        del state._log[:]
        state.set_parent_trees([(with_a_id, tree.branch.repository.revision_tree(with_a_id))], [])
        entry = state._get_entry(0, path_utf8=b'a')
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6', link_or_sha1)
        self.assertEqual([('is_exec', mode, False), ('sha1', b'a')], state._log)
        self.assertEqual((b'f', link_or_sha1, 14, False, packed_stat), entry[1][0])
        del state._log[:]
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6', link_or_sha1)
        self.assertEqual([], state._log)
        self.assertEqual((b'f', link_or_sha1, 14, False, packed_stat), entry[1][0])

    def test_update_entry_symlink(self):
        """Update entry should read symlinks."""
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        state, entry = self.get_state_with_a()
        state.save()
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        os.symlink('target', 'a')
        state.adjust_time(-10)
        stat_value = os.lstat('a')
        packed_stat = dirstate.pack_stat(stat_value)
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(b'target', link_or_sha1)
        self.assertEqual([('read_link', b'a', b'')], state._log)
        self.assertEqual([(b'l', b'', 6, False, dirstate.DirState.NULLSTAT)], entry[1])
        self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
        del state._log[:]
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(b'target', link_or_sha1)
        self.assertEqual([('read_link', b'a', b'')], state._log)
        self.assertEqual([(b'l', b'', 6, False, dirstate.DirState.NULLSTAT)], entry[1])
        state.save()
        state.adjust_time(+20)
        del state._log[:]
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        self.assertEqual(b'target', link_or_sha1)
        self.assertEqual([('read_link', b'a', b'')], state._log)
        self.assertEqual([(b'l', b'target', 6, False, packed_stat)], entry[1])
        del state._log[:]
        self.assertEqual([], state._log)
        link_or_sha1 = self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual(b'target', link_or_sha1)
        self.assertEqual([(b'l', b'target', 6, False, packed_stat)], entry[1])

    def do_update_entry(self, state, entry, abspath):
        stat_value = os.lstat(abspath)
        return self.update_entry(state, entry, abspath, stat_value)

    def test_update_entry_dir(self):
        state, entry = self.get_state_with_a()
        self.build_tree(['a/'])
        self.assertIs(None, self.do_update_entry(state, entry, b'a'))

    def test_update_entry_dir_unchanged(self):
        state, entry = self.get_state_with_a()
        self.build_tree(['a/'])
        state.adjust_time(+20)
        self.assertIs(None, self.do_update_entry(state, entry, b'a'))
        self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
        state.save()
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        self.assertIs(None, self.do_update_entry(state, entry, b'a'))
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        t = time.time() - 100.0
        try:
            os.utime('a', (t, t))
        except OSError:
            raise tests.TestSkipped("can't update mtime of a dir on FAT")
        saved_packed_stat = entry[1][0][-1]
        self.assertIs(None, self.do_update_entry(state, entry, b'a'))
        self.assertNotEqual(saved_packed_stat, entry[1][0][-1])
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)

    def test_update_entry_file_unchanged(self):
        state, _ = self.get_state_with_a()
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.build_tree(['tree/a'])
        tree.add(['a'], ids=[b'a-id'])
        with_a_id = tree.commit('witha')
        self.addCleanup(tree.unlock)
        state.set_parent_trees([(with_a_id, tree.branch.repository.revision_tree(with_a_id))], [])
        entry = state._get_entry(0, path_utf8=b'a')
        self.build_tree(['a'])
        sha1sum = b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6'
        state.adjust_time(+20)
        self.assertEqual(sha1sum, self.do_update_entry(state, entry, b'a'))
        self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
        state.save()
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
        self.assertEqual(sha1sum, self.do_update_entry(state, entry, b'a'))
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)

    def test_update_entry_tree_reference(self):
        state = test_dirstate.InstrumentedDirState.initialize('dirstate')
        self.addCleanup(state.unlock)
        state.add('r', b'r-id', 'tree-reference', None, b'')
        self.build_tree(['r/'])
        entry = state._get_entry(0, path_utf8=b'r')
        self.do_update_entry(state, entry, 'r')
        entry = state._get_entry(0, path_utf8=b'r')
        self.assertEqual(b't', entry[1][0][0])

    def create_and_test_file(self, state, entry):
        """Create a file at 'a' and verify the state finds it during update.

        The state should already be versioning *something* at 'a'. This makes
        sure that state.update_entry recognizes it as a file.
        """
        self.build_tree(['a'])
        stat_value = os.lstat('a')
        packed_stat = dirstate.pack_stat(stat_value)
        link_or_sha1 = self.do_update_entry(state, entry, abspath='a')
        self.assertEqual(None, link_or_sha1)
        self.assertEqual([(b'f', b'', 14, False, dirstate.DirState.NULLSTAT)], entry[1])
        return packed_stat

    def create_and_test_dir(self, state, entry):
        """Create a directory at 'a' and verify the state finds it.

        The state should already be versioning *something* at 'a'. This makes
        sure that state.update_entry recognizes it as a directory.
        """
        self.build_tree(['a/'])
        stat_value = os.lstat('a')
        packed_stat = dirstate.pack_stat(stat_value)
        link_or_sha1 = self.do_update_entry(state, entry, abspath=b'a')
        self.assertIs(None, link_or_sha1)
        self.assertEqual([(b'd', b'', 0, False, packed_stat)], entry[1])
        return packed_stat

    def create_and_test_symlink(self, state, entry):
        """Create a symlink at 'a' and verify the state finds it.

        The state should already be versioning *something* at 'a'. This makes
        sure that state.update_entry recognizes it as a symlink.

        This should not be called if this platform does not have symlink
        support.
        """
        os.symlink('path/to/foo', 'a')
        stat_value = os.lstat('a')
        packed_stat = dirstate.pack_stat(stat_value)
        link_or_sha1 = self.do_update_entry(state, entry, abspath=b'a')
        self.assertEqual(b'path/to/foo', link_or_sha1)
        self.assertEqual([(b'l', b'path/to/foo', 11, False, packed_stat)], entry[1])
        return packed_stat

    def test_update_file_to_dir(self):
        """If a file changes to a directory we return None for the sha.
        We also update the inventory record.
        """
        state, entry = self.get_state_with_a()
        state.adjust_time(+10)
        self.create_and_test_file(state, entry)
        os.remove('a')
        self.create_and_test_dir(state, entry)

    def test_update_file_to_symlink(self):
        """File becomes a symlink"""
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        state, entry = self.get_state_with_a()
        state.adjust_time(+10)
        self.create_and_test_file(state, entry)
        os.remove('a')
        self.create_and_test_symlink(state, entry)

    def test_update_dir_to_file(self):
        """Directory becoming a file updates the entry."""
        state, entry = self.get_state_with_a()
        state.adjust_time(+10)
        self.create_and_test_dir(state, entry)
        os.rmdir('a')
        self.create_and_test_file(state, entry)

    def test_update_dir_to_symlink(self):
        """Directory becomes a symlink"""
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        state, entry = self.get_state_with_a()
        state.adjust_time(+10)
        self.create_and_test_dir(state, entry)
        os.rmdir('a')
        self.create_and_test_symlink(state, entry)

    def test_update_symlink_to_file(self):
        """Symlink becomes a file"""
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        state, entry = self.get_state_with_a()
        state.adjust_time(+10)
        self.create_and_test_symlink(state, entry)
        os.remove('a')
        self.create_and_test_file(state, entry)

    def test_update_symlink_to_dir(self):
        """Symlink becomes a directory"""
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        state, entry = self.get_state_with_a()
        state.adjust_time(+10)
        self.create_and_test_symlink(state, entry)
        os.remove('a')
        self.create_and_test_dir(state, entry)

    def test__is_executable_win32(self):
        state, entry = self.get_state_with_a()
        self.build_tree(['a'])
        state._use_filesystem_for_exec = False
        entry[1][0] = (b'f', b'', 0, True, dirstate.DirState.NULLSTAT)
        stat_value = os.lstat('a')
        packed_stat = dirstate.pack_stat(stat_value)
        state.adjust_time(-10)
        self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual([(b'f', b'', 14, True, dirstate.DirState.NULLSTAT)], entry[1])
        state.adjust_time(+20)
        digest = b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6'
        self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
        self.assertEqual([(b'f', b'', 14, True, dirstate.DirState.NULLSTAT)], entry[1])

    def _prepare_tree(self):
        text = b'Hello World\n'
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a file', text)])
        tree.add('a file', ids=b'a-file-id')
        tree.commit('first')
        return (tree, text)

    def test_sha1provider_sha1_used(self):
        tree, text = self._prepare_tree()
        state = dirstate.DirState.from_tree(tree, 'dirstate', UppercaseSHA1Provider())
        self.addCleanup(state.unlock)
        expected_sha = osutils.sha_string(text.upper() + b'foo')
        entry = state._get_entry(0, path_utf8=b'a file')
        self.assertNotEqual((None, None), entry)
        state._sha_cutoff_time()
        state._cutoff_time += 10
        sha1 = self.update_entry(state, entry, 'tree/a file', os.lstat('tree/a file'))
        self.assertEqual(expected_sha, sha1)

    def test_sha1provider_stat_and_sha1_used(self):
        tree, text = self._prepare_tree()
        tree.lock_write()
        self.addCleanup(tree.unlock)
        state = tree._current_dirstate()
        state._sha1_provider = UppercaseSHA1Provider()
        file_ids_changed = [change.file_id for change in tree.iter_changes(tree.basis_tree())]
        self.assertEqual([b'a-file-id'], file_ids_changed)