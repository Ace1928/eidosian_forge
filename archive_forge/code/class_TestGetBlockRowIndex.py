import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestGetBlockRowIndex(TestCaseWithDirState):

    def assertBlockRowIndexEqual(self, block_index, row_index, dir_present, file_present, state, dirname, basename, tree_index):
        self.assertEqual((block_index, row_index, dir_present, file_present), state._get_block_entry_index(dirname, basename, tree_index))
        if dir_present:
            block = state._dirblocks[block_index]
            self.assertEqual(dirname, block[0])
        if dir_present and file_present:
            row = state._dirblocks[block_index][1][row_index]
            self.assertEqual(dirname, row[0][0])
            self.assertEqual(basename, row[0][1])

    def test_simple_structure(self):
        state = self.create_dirstate_with_root_and_subdir()
        self.addCleanup(state.unlock)
        self.assertBlockRowIndexEqual(1, 0, True, True, state, b'', b'subdir', 0)
        self.assertBlockRowIndexEqual(1, 0, True, False, state, b'', b'bdir', 0)
        self.assertBlockRowIndexEqual(1, 1, True, False, state, b'', b'zdir', 0)
        self.assertBlockRowIndexEqual(2, 0, False, False, state, b'a', b'foo', 0)
        self.assertBlockRowIndexEqual(2, 0, False, False, state, b'subdir', b'foo', 0)

    def test_complex_structure_exists(self):
        state = self.create_complex_dirstate()
        self.addCleanup(state.unlock)
        self.assertBlockRowIndexEqual(0, 0, True, True, state, b'', b'', 0)
        self.assertBlockRowIndexEqual(1, 0, True, True, state, b'', b'a', 0)
        self.assertBlockRowIndexEqual(1, 1, True, True, state, b'', b'b', 0)
        self.assertBlockRowIndexEqual(1, 2, True, True, state, b'', b'c', 0)
        self.assertBlockRowIndexEqual(1, 3, True, True, state, b'', b'd', 0)
        self.assertBlockRowIndexEqual(2, 0, True, True, state, b'a', b'e', 0)
        self.assertBlockRowIndexEqual(2, 1, True, True, state, b'a', b'f', 0)
        self.assertBlockRowIndexEqual(3, 0, True, True, state, b'b', b'g', 0)
        self.assertBlockRowIndexEqual(3, 1, True, True, state, b'b', b'h\xc3\xa5', 0)

    def test_complex_structure_missing(self):
        state = self.create_complex_dirstate()
        self.addCleanup(state.unlock)
        self.assertBlockRowIndexEqual(0, 0, True, True, state, b'', b'', 0)
        self.assertBlockRowIndexEqual(1, 0, True, False, state, b'', b'_', 0)
        self.assertBlockRowIndexEqual(1, 1, True, False, state, b'', b'aa', 0)
        self.assertBlockRowIndexEqual(1, 4, True, False, state, b'', b'h\xc3\xa5', 0)
        self.assertBlockRowIndexEqual(2, 0, False, False, state, b'_', b'a', 0)
        self.assertBlockRowIndexEqual(3, 0, False, False, state, b'aa', b'a', 0)
        self.assertBlockRowIndexEqual(4, 0, False, False, state, b'bb', b'a', 0)
        self.assertBlockRowIndexEqual(3, 0, False, False, state, b'a/e', b'a', 0)
        self.assertBlockRowIndexEqual(4, 0, False, False, state, b'e', b'a', 0)