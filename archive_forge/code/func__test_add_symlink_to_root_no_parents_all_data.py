import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
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