import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestProcessEntry(test_dirstate.TestCaseWithDirState):
    scenarios = multiply_scenarios(dir_reader_scenarios(), pe_scenarios)
    _process_entry = None

    def setUp(self):
        super().setUp()
        self.overrideAttr(dirstate, '_process_entry', self._process_entry)

    def assertChangedFileIds(self, expected, tree):
        with tree.lock_read():
            file_ids = [info.file_id for info in tree.iter_changes(tree.basis_tree())]
        self.assertEqual(sorted(expected), sorted(file_ids))

    def test_exceptions_raised(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file', 'tree/dir/', 'tree/dir/sub', 'tree/dir2/', 'tree/dir2/sub2'])
        tree.add(['file', 'dir', 'dir/sub', 'dir2', 'dir2/sub2'])
        tree.commit('first commit')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        basis_tree = tree.basis_tree()

        def is_inside_raises(*args, **kwargs):
            raise RuntimeError('stop this')
        self.overrideAttr(dirstate, 'is_inside', is_inside_raises)
        try:
            from .. import _dirstate_helpers_pyx
        except ImportError:
            pass
        else:
            self.overrideAttr(_dirstate_helpers_pyx, 'is_inside', is_inside_raises)
        self.overrideAttr(osutils, 'is_inside', is_inside_raises)
        self.assertListRaises(RuntimeError, tree.iter_changes, basis_tree)

    def test_simple_changes(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file'])
        tree.add(['file'], ids=[b'file-id'])
        self.assertChangedFileIds([tree.path2id(''), b'file-id'], tree)
        tree.commit('one')
        self.assertChangedFileIds([], tree)

    def test_sha1provider_stat_and_sha1_used(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file'])
        tree.add(['file'], ids=[b'file-id'])
        tree.commit('one')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        state = tree._current_dirstate()
        state._sha1_provider = UppercaseSHA1Provider()
        self.assertChangedFileIds([b'file-id'], tree)