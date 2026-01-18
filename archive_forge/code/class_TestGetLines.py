import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestGetLines(TestCaseWithDirState):

    def test_get_line_with_2_rows(self):
        state = self.create_dirstate_with_root_and_subdir()
        try:
            self.assertEqual([b'#bazaar dirstate flat format 3\n', b'crc32: 41262208\n', b'num_entries: 2\n', b'0\x00\n\x000\x00\n\x00\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00\n\x00\x00subdir\x00subdir-id\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00\n\x00'], state.get_lines())
        finally:
            state.unlock()

    def test_entry_to_line(self):
        state = self.create_dirstate_with_root()
        try:
            self.assertEqual(b'\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk', state._entry_to_line(state._dirblocks[0][1][0]))
        finally:
            state.unlock()

    def test_entry_to_line_with_parent(self):
        packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
        root_entry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat), (b'a', b'dirname/basename', 0, False, b'')])
        state = dirstate.DirState.initialize('dirstate')
        try:
            self.assertEqual(b'\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00a\x00dirname/basename\x000\x00n\x00', state._entry_to_line(root_entry))
        finally:
            state.unlock()

    def test_entry_to_line_with_two_parents_at_different_paths(self):
        packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
        root_entry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat), (b'd', b'', 0, False, b'rev_id'), (b'a', b'dirname/basename', 0, False, b'')])
        state = dirstate.DirState.initialize('dirstate')
        try:
            self.assertEqual(b'\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00d\x00\x000\x00n\x00rev_id\x00a\x00dirname/basename\x000\x00n\x00', state._entry_to_line(root_entry))
        finally:
            state.unlock()

    def test_iter_entries(self):
        packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
        dirblocks = []
        root_entries = [((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat)])]
        dirblocks.append(('', root_entries))
        subdir_entry = ((b'', b'subdir', b'subdir-id'), [(b'd', b'', 0, False, packed_stat)])
        afile_entry = ((b'', b'afile', b'afile-id'), [(b'f', b'sha1value', 34, False, packed_stat)])
        dirblocks.append(('', [subdir_entry, afile_entry]))
        file_entry2 = ((b'subdir', b'2file', b'2file-id'), [(b'f', b'sha1value', 23, False, packed_stat)])
        dirblocks.append(('subdir', [file_entry2]))
        state = dirstate.DirState.initialize('dirstate')
        try:
            state._set_data([], dirblocks)
            expected_entries = [root_entries[0], subdir_entry, afile_entry, file_entry2]
            self.assertEqual(expected_entries, list(state._iter_entries()))
        finally:
            state.unlock()