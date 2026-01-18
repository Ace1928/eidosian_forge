import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestDirstateTreeReference(TestCaseWithDirState):

    def test_reference_revision_is_none(self):
        tree = self.make_branch_and_tree('tree', format='development-subtree')
        subtree = self.make_branch_and_tree('tree/subtree', format='development-subtree')
        subtree.set_root_id(b'subtree')
        tree.add_reference(subtree)
        tree.add('subtree')
        state = dirstate.DirState.from_tree(tree, 'dirstate')
        key = (b'', b'subtree', b'subtree')
        expected = (b'', [(key, [(b't', b'', 0, False, b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')])])
        try:
            self.assertEqual(expected, state._find_block(key))
        finally:
            state.unlock()