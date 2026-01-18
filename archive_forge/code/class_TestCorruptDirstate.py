import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
class TestCorruptDirstate(TestCaseWithTransport):
    """Tests for how we handle when the dirstate has been corrupted."""

    def create_wt4(self):
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        control.create_repository()
        control.create_branch()
        tree = workingtree_4.WorkingTreeFormat4().initialize(control)
        return tree

    def test_invalid_rename(self):
        tree = self.create_wt4()
        with tree.lock_write():
            tree.commit('init')
            state = tree.current_dirstate()
            state._read_dirblocks_if_needed()
            state._dirblocks[1][1].append(((b'', b'foo', b'foo-id'), [(b'f', b'', 0, False, b''), (b'r', b'bar', 0, False, b'')]))
            self.assertListRaises(dirstate.DirstateCorrupt, tree.iter_changes, tree.basis_tree())

    def get_simple_dirblocks(self, state):
        """Extract the simple information from the DirState.

        This returns the dirblocks, only with the sha1sum and stat details
        filtered out.
        """
        simple_blocks = []
        for block in state._dirblocks:
            simple_block = (block[0], [])
            for entry in block[1]:
                simple_block[1].append((entry[0], [i[0] for i in entry[1]]))
            simple_blocks.append(simple_block)
        return simple_blocks

    def test_update_basis_with_invalid_delta(self):
        """When given an invalid delta, it should abort, and not be saved."""
        self.build_tree(['dir/', 'dir/file'])
        tree = self.create_wt4()
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
        first_revision_id = tree.commit('init')
        root_id = tree.path2id('')
        state = tree.current_dirstate()
        state._read_dirblocks_if_needed()
        self.assertEqual([(b'', [((b'', b'', root_id), [b'd', b'd'])]), (b'', [((b'', b'dir', b'dir-id'), [b'd', b'd'])]), (b'dir', [((b'dir', b'file', b'file-id'), [b'f', b'f'])])], self.get_simple_dirblocks(state))
        tree.remove(['dir/file'])
        self.assertEqual([(b'', [((b'', b'', root_id), [b'd', b'd'])]), (b'', [((b'', b'dir', b'dir-id'), [b'd', b'd'])]), (b'dir', [((b'dir', b'file', b'file-id'), [b'a', b'f'])])], self.get_simple_dirblocks(state))
        tree.flush()
        new_dir = inventory.InventoryDirectory(b'dir-id', 'new-dir', root_id)
        new_dir.revision = b'new-revision-id'
        new_file = inventory.InventoryFile(b'file-id', 'new-file', root_id)
        new_file.revision = b'new-revision-id'
        self.assertRaises(errors.InconsistentDelta, tree.update_basis_by_delta, b'new-revision-id', [('dir', 'new-dir', b'dir-id', new_dir), ('dir/file', 'new-dir/new-file', b'file-id', new_file)])
        del state
        tree.unlock()
        tree.lock_read()
        self.assertEqual(first_revision_id, tree.last_revision())
        state = tree.current_dirstate()
        state._read_dirblocks_if_needed()
        self.assertEqual([(b'', [((b'', b'', root_id), [b'd', b'd'])]), (b'', [((b'', b'dir', b'dir-id'), [b'd', b'd'])]), (b'dir', [((b'dir', b'file', b'file-id'), [b'a', b'f'])])], self.get_simple_dirblocks(state))