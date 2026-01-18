import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestReadDirblocks(test_dirstate.TestCaseWithDirState):
    """Test an implementation of _read_dirblocks()

    _read_dirblocks() reads in all of the dirblock information from the disk
    file.

    Child test cases can override ``get_read_dirblocks`` to test a specific
    implementation.
    """

    def get_read_dirblocks(self):
        from .._dirstate_helpers_py import _read_dirblocks
        return _read_dirblocks

    def test_smoketest(self):
        """Make sure that we can create and read back a simple file."""
        tree, state, expected = self.create_basic_dirstate()
        del tree
        state._read_header_if_needed()
        self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._dirblock_state)
        read_dirblocks = self.get_read_dirblocks()
        read_dirblocks(state)
        self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)

    def test_trailing_garbage(self):
        tree, state, expected = self.create_basic_dirstate()
        state.unlock()
        f = open('dirstate', 'ab')
        try:
            f.write(b'bogus\n')
        finally:
            f.close()
            state.lock_read()
        e = self.assertRaises(dirstate.DirstateCorrupt, state._read_dirblocks_if_needed)
        self.assertContainsRe(str(e), 'bogus')