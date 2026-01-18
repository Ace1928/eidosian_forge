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
class FindMissingBzrRevidsTests(TestCase):

    def _find_missing(self, ancestry, want, have):
        return _find_missing_bzr_revids(Graph(DictParentsProvider(ancestry)), set(want), set(have))

    def test_simple(self):
        self.assertEqual(set(), self._find_missing({}, [], []))

    def test_up_to_date(self):
        self.assertEqual(set(), self._find_missing({'a': ['b']}, ['a'], ['a']))

    def test_one_missing(self):
        self.assertEqual({'a'}, self._find_missing({'a': ['b']}, ['a'], ['b']))

    def test_two_missing(self):
        self.assertEqual({'a', 'b'}, self._find_missing({'a': ['b'], 'b': ['c']}, ['a'], ['c']))

    def test_two_missing_history(self):
        self.assertEqual({'a', 'b'}, self._find_missing({'a': ['b'], 'b': ['c'], 'c': ['d']}, ['a'], ['c']))