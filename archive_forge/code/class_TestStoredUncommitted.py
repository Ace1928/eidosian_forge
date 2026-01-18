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
class TestStoredUncommitted(TestCaseWithTransport):

    def store_uncommitted(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('get root in there')
        self.build_tree_contents([('tree/file', b'content')])
        tree.add('file', ids=b'file-id')
        tree.store_uncommitted()
        return tree

    def test_store_uncommitted(self):
        self.store_uncommitted()
        self.assertPathDoesNotExist('tree/file')

    def test_store_uncommitted_no_change(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('get root in there')
        tree.store_uncommitted()
        self.assertIs(None, tree.branch.get_unshelver(tree))

    def test_restore_uncommitted(self):
        with write_locked(self.store_uncommitted()) as tree:
            tree.restore_uncommitted()
            self.assertPathExists('tree/file')
            self.assertIs(None, tree.branch.get_unshelver(tree))

    def test_restore_uncommitted_none(self):
        tree = self.make_branch_and_tree('tree')
        tree.restore_uncommitted()