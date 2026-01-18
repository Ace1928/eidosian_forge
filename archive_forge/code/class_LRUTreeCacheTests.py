import os
import shutil
import stat
from dulwich.objects import Blob, Tree
from ...branchbuilder import BranchBuilder
from ...bzr.inventory import InventoryDirectory, InventoryFile
from ...errors import NoSuchRevision
from ...graph import DictParentsProvider, Graph
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import SymlinkFeature
from ..cache import DictGitShaMap
from ..object_store import (BazaarObjectStore, LRUTreeCache,
class LRUTreeCacheTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.branch = self.make_branch('.')
        self.branch.lock_write()
        self.addCleanup(self.branch.unlock)
        self.cache = LRUTreeCache(self.branch.repository)

    def test_get_not_present(self):
        self.assertRaises(NoSuchRevision, self.cache.revision_tree, 'unknown')

    def test_revision_trees(self):
        self.assertRaises(NoSuchRevision, self.cache.revision_trees, ['unknown', 'la'])

    def test_iter_revision_trees(self):
        self.assertRaises(NoSuchRevision, self.cache.iter_revision_trees, ['unknown', 'la'])

    def test_get(self):
        bb = BranchBuilder(branch=self.branch)
        bb.start_series()
        revid = bb.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\nd\ne\n'))])
        bb.finish_series()
        tree = self.cache.revision_tree(revid)
        self.assertEqual(revid, tree.get_revision_id())