import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestCaseWithDirState(tests.TestCaseWithTransport):
    """Helper functions for creating DirState objects with various content."""
    scenarios = test_osutils.dir_reader_scenarios()
    _dir_reader_class = None
    _native_to_unicode = None

    def setUp(self):
        super().setUp()
        self.overrideAttr(osutils, '_selected_dir_reader', self._dir_reader_class())

    def create_empty_dirstate(self):
        """Return a locked but empty dirstate"""
        state = dirstate.DirState.initialize('dirstate')
        return state

    def create_dirstate_with_root(self):
        """Return a write-locked state with a single root entry."""
        packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
        root_entry_direntry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat)])
        dirblocks = []
        dirblocks.append((b'', [root_entry_direntry]))
        dirblocks.append((b'', []))
        state = self.create_empty_dirstate()
        try:
            state._set_data([], dirblocks)
            state._validate()
        except:
            state.unlock()
            raise
        return state

    def create_dirstate_with_root_and_subdir(self):
        """Return a locked DirState with a root and a subdir"""
        packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
        subdir_entry = ((b'', b'subdir', b'subdir-id'), [(b'd', b'', 0, False, packed_stat)])
        state = self.create_dirstate_with_root()
        try:
            dirblocks = list(state._dirblocks)
            dirblocks[1][1].append(subdir_entry)
            state._set_data([], dirblocks)
        except:
            state.unlock()
            raise
        return state

    def create_complex_dirstate(self):
        """This dirstate contains multiple files and directories.

         /        a-root-value
         a/       a-dir
         b/       b-dir
         c        c-file
         d        d-file
         a/e/     e-dir
         a/f      f-file
         b/g      g-file
         b/hÃ¥  h-Ã¥-file  #This is u'å' encoded into utf-8

        Notice that a/e is an empty directory.

        :return: The dirstate, still write-locked.
        """
        packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
        null_sha = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        root_entry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat)])
        a_entry = ((b'', b'a', b'a-dir'), [(b'd', b'', 0, False, packed_stat)])
        b_entry = ((b'', b'b', b'b-dir'), [(b'd', b'', 0, False, packed_stat)])
        c_entry = ((b'', b'c', b'c-file'), [(b'f', null_sha, 10, False, packed_stat)])
        d_entry = ((b'', b'd', b'd-file'), [(b'f', null_sha, 20, False, packed_stat)])
        e_entry = ((b'a', b'e', b'e-dir'), [(b'd', b'', 0, False, packed_stat)])
        f_entry = ((b'a', b'f', b'f-file'), [(b'f', null_sha, 30, False, packed_stat)])
        g_entry = ((b'b', b'g', b'g-file'), [(b'f', null_sha, 30, False, packed_stat)])
        h_entry = ((b'b', b'h\xc3\xa5', b'h-\xc3\xa5-file'), [(b'f', null_sha, 40, False, packed_stat)])
        dirblocks = []
        dirblocks.append((b'', [root_entry]))
        dirblocks.append((b'', [a_entry, b_entry, c_entry, d_entry]))
        dirblocks.append((b'a', [e_entry, f_entry]))
        dirblocks.append((b'b', [g_entry, h_entry]))
        state = dirstate.DirState.initialize('dirstate')
        state._validate()
        try:
            state._set_data([], dirblocks)
        except:
            state.unlock()
            raise
        return state

    def check_state_with_reopen(self, expected_result, state):
        """Check that state has current state expected_result.

        This will check the current state, open the file anew and check it
        again.
        This function expects the current state to be locked for writing, and
        will unlock it before re-opening.
        This is required because we can't open a lock_read() while something
        else has a lock_write().
            write => mutually exclusive lock
            read => shared lock
        """
        self.assertTrue(state._lock_token is not None)
        try:
            self.assertEqual(expected_result[0], state.get_parent_ids())
            self.assertEqual([], state.get_ghosts())
            self.assertEqual(expected_result[1], list(state._iter_entries()))
            state.save()
        finally:
            state.unlock()
        del state
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            self.assertEqual(expected_result[1], list(state._iter_entries()))
        finally:
            state.unlock()

    def create_basic_dirstate(self):
        """Create a dirstate with a few files and directories.

            a
            b/
              c
              d/
                e
            b-c
            f
        """
        tree = self.make_branch_and_tree('tree')
        paths = ['a', 'b/', 'b/c', 'b/d/', 'b/d/e', 'b-c', 'f']
        file_ids = [b'a-id', b'b-id', b'c-id', b'd-id', b'e-id', b'b-c-id', b'f-id']
        self.build_tree(['tree/' + p for p in paths])
        tree.set_root_id(b'TREE_ROOT')
        tree.add([p.rstrip('/') for p in paths], ids=file_ids)
        tree.commit('initial', rev_id=b'rev-1')
        revision_id = b'rev-1'
        t = self.get_transport('tree')
        a_text = t.get_bytes('a')
        a_sha = osutils.sha_string(a_text)
        a_len = len(a_text)
        c_text = t.get_bytes('b/c')
        c_sha = osutils.sha_string(c_text)
        c_len = len(c_text)
        e_text = t.get_bytes('b/d/e')
        e_sha = osutils.sha_string(e_text)
        e_len = len(e_text)
        b_c_text = t.get_bytes('b-c')
        b_c_sha = osutils.sha_string(b_c_text)
        b_c_len = len(b_c_text)
        f_text = t.get_bytes('f')
        f_sha = osutils.sha_string(f_text)
        f_len = len(f_text)
        null_stat = dirstate.DirState.NULLSTAT
        expected = {b'': ((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, null_stat), (b'd', b'', 0, False, revision_id)]), b'a': ((b'', b'a', b'a-id'), [(b'f', b'', 0, False, null_stat), (b'f', a_sha, a_len, False, revision_id)]), b'b': ((b'', b'b', b'b-id'), [(b'd', b'', 0, False, null_stat), (b'd', b'', 0, False, revision_id)]), b'b/c': ((b'b', b'c', b'c-id'), [(b'f', b'', 0, False, null_stat), (b'f', c_sha, c_len, False, revision_id)]), b'b/d': ((b'b', b'd', b'd-id'), [(b'd', b'', 0, False, null_stat), (b'd', b'', 0, False, revision_id)]), b'b/d/e': ((b'b/d', b'e', b'e-id'), [(b'f', b'', 0, False, null_stat), (b'f', e_sha, e_len, False, revision_id)]), b'b-c': ((b'', b'b-c', b'b-c-id'), [(b'f', b'', 0, False, null_stat), (b'f', b_c_sha, b_c_len, False, revision_id)]), b'f': ((b'', b'f', b'f-id'), [(b'f', b'', 0, False, null_stat), (b'f', f_sha, f_len, False, revision_id)])}
        state = dirstate.DirState.from_tree(tree, 'dirstate')
        try:
            state.save()
        finally:
            state.unlock()
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        self.addCleanup(state.unlock)
        self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._dirblock_state)
        state._bisect_page_size = 200
        return (tree, state, expected)

    def create_duplicated_dirstate(self):
        """Create a dirstate with a deleted and added entries.

        This grabs a basic_dirstate, and then removes and re adds every entry
        with a new file id.
        """
        tree, state, expected = self.create_basic_dirstate()
        tree.unversion(['f', 'b-c', 'b/d/e', 'b/d', 'b/c', 'b', 'a'])
        tree.add(['a', 'b', 'b/c', 'b/d', 'b/d/e', 'b-c', 'f'], ids=[b'a-id2', b'b-id2', b'c-id2', b'd-id2', b'e-id2', b'b-c-id2', b'f-id2'])
        for path in [b'a', b'b', b'b/c', b'b/d', b'b/d/e', b'b-c', b'f']:
            orig = expected[path]
            path2 = path + b'2'
            expected[path] = (orig[0], [dirstate.DirState.NULL_PARENT_DETAILS, orig[1][1]])
            new_key = (orig[0][0], orig[0][1], orig[0][2] + b'2')
            expected[path2] = (new_key, [orig[1][0], dirstate.DirState.NULL_PARENT_DETAILS])
        state.unlock()
        try:
            new_state = dirstate.DirState.from_tree(tree, 'dirstate')
            try:
                new_state.save()
            finally:
                new_state.unlock()
        finally:
            state.lock_read()
        return (tree, state, expected)

    def create_renamed_dirstate(self):
        """Create a dirstate with a few internal renames.

        This takes the basic dirstate, and moves the paths around.
        """
        tree, state, expected = self.create_basic_dirstate()
        tree.rename_one('a', 'b/g')
        tree.rename_one('b/d', 'h')
        old_a = expected[b'a']
        expected[b'a'] = (old_a[0], [(b'r', b'b/g', 0, False, b''), old_a[1][1]])
        expected[b'b/g'] = ((b'b', b'g', b'a-id'), [old_a[1][0], (b'r', b'a', 0, False, b'')])
        old_d = expected[b'b/d']
        expected[b'b/d'] = (old_d[0], [(b'r', b'h', 0, False, b''), old_d[1][1]])
        expected[b'h'] = ((b'', b'h', b'd-id'), [old_d[1][0], (b'r', b'b/d', 0, False, b'')])
        old_e = expected[b'b/d/e']
        expected[b'b/d/e'] = (old_e[0], [(b'r', b'h/e', 0, False, b''), old_e[1][1]])
        expected[b'h/e'] = ((b'h', b'e', b'e-id'), [old_e[1][0], (b'r', b'b/d/e', 0, False, b'')])
        state.unlock()
        try:
            new_state = dirstate.DirState.from_tree(tree, 'dirstate')
            try:
                new_state.save()
            finally:
                new_state.unlock()
        finally:
            state.lock_read()
        return (tree, state, expected)