import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
class TestWorkingTreeFormat3(TestCaseWithTransport):
    """Tests specific to WorkingTreeFormat3."""

    def test_disk_layout(self):
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        control.create_repository()
        control.create_branch()
        workingtree_3.WorkingTreeFormat3().initialize(control)
        t = control.get_workingtree_transport(None)
        self.assertEqualDiff(b'Bazaar-NG Working Tree format 3', t.get('format').read())
        self.assertEqualDiff(t.get('inventory').read(), b'<inventory format="5">\n</inventory>\n')
        self.assertEqualDiff(b'### bzr hashcache v5\n', t.get('stat-cache').read())
        self.assertFalse(t.has('inventory.basis'))
        self.assertFalse(t.has('last-revision'))

    def test_uses_lockdir(self):
        """WorkingTreeFormat3 uses its own LockDir:

            - lock is a directory
            - when the WorkingTree is locked, LockDir can see that
        """
        t = self.get_transport()
        url = self.get_url()
        dir = bzrdir.BzrDirMetaFormat1().initialize(url)
        dir.create_repository()
        dir.create_branch()
        try:
            tree = workingtree_3.WorkingTreeFormat3().initialize(dir)
        except errors.NotLocalUrl:
            raise TestSkipped('Not a local URL')
        self.assertIsDirectory('.bzr', t)
        self.assertIsDirectory('.bzr/checkout', t)
        self.assertIsDirectory('.bzr/checkout/lock', t)
        our_lock = LockDir(t, '.bzr/checkout/lock')
        self.assertEqual(our_lock.peek(), None)
        with tree.lock_write():
            self.assertTrue(our_lock.peek())
        self.assertEqual(our_lock.peek(), None)

    def test_missing_pending_merges(self):
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        control.create_repository()
        control.create_branch()
        tree = workingtree_3.WorkingTreeFormat3().initialize(control)
        tree._transport.delete('pending-merges')
        self.assertEqual([], tree.get_parent_ids())