import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestGetEntry(TestCaseWithDirState):

    def assertEntryEqual(self, dirname, basename, file_id, state, path, index):
        """Check that the right entry is returned for a request to getEntry."""
        entry = state._get_entry(index, path_utf8=path)
        if file_id is None:
            self.assertEqual((None, None), entry)
        else:
            cur = entry[0]
            self.assertEqual((dirname, basename, file_id), cur[:3])

    def test_simple_structure(self):
        state = self.create_dirstate_with_root_and_subdir()
        self.addCleanup(state.unlock)
        self.assertEntryEqual(b'', b'', b'a-root-value', state, b'', 0)
        self.assertEntryEqual(b'', b'subdir', b'subdir-id', state, b'subdir', 0)
        self.assertEntryEqual(None, None, None, state, b'missing', 0)
        self.assertEntryEqual(None, None, None, state, b'missing/foo', 0)
        self.assertEntryEqual(None, None, None, state, b'subdir/foo', 0)

    def test_complex_structure_exists(self):
        state = self.create_complex_dirstate()
        self.addCleanup(state.unlock)
        self.assertEntryEqual(b'', b'', b'a-root-value', state, b'', 0)
        self.assertEntryEqual(b'', b'a', b'a-dir', state, b'a', 0)
        self.assertEntryEqual(b'', b'b', b'b-dir', state, b'b', 0)
        self.assertEntryEqual(b'', b'c', b'c-file', state, b'c', 0)
        self.assertEntryEqual(b'', b'd', b'd-file', state, b'd', 0)
        self.assertEntryEqual(b'a', b'e', b'e-dir', state, b'a/e', 0)
        self.assertEntryEqual(b'a', b'f', b'f-file', state, b'a/f', 0)
        self.assertEntryEqual(b'b', b'g', b'g-file', state, b'b/g', 0)
        self.assertEntryEqual(b'b', b'h\xc3\xa5', b'h-\xc3\xa5-file', state, b'b/h\xc3\xa5', 0)

    def test_complex_structure_missing(self):
        state = self.create_complex_dirstate()
        self.addCleanup(state.unlock)
        self.assertEntryEqual(None, None, None, state, b'_', 0)
        self.assertEntryEqual(None, None, None, state, b'_\xc3\xa5', 0)
        self.assertEntryEqual(None, None, None, state, b'a/b', 0)
        self.assertEntryEqual(None, None, None, state, b'c/d', 0)

    def test_get_entry_uninitialized(self):
        """Calling get_entry will load data if it needs to"""
        state = self.create_dirstate_with_root()
        try:
            state.save()
        finally:
            state.unlock()
        del state
        state = dirstate.DirState.on_file('dirstate')
        state.lock_read()
        try:
            self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._header_state)
            self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._dirblock_state)
            self.assertEntryEqual(b'', b'', b'a-root-value', state, b'', 0)
        finally:
            state.unlock()