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
class DirectoryToTreeTests(TestCase):

    def test_empty(self):
        t = directory_to_tree('', [], None, {}, None, allow_empty=False)
        self.assertEqual(None, t)

    def test_empty_dir(self):
        child_ie = InventoryDirectory(b'bar', 'bar', b'bar')
        t = directory_to_tree('', [child_ie], lambda p, x: None, {}, None, allow_empty=False)
        self.assertEqual(None, t)

    def test_empty_dir_dummy_files(self):
        child_ie = InventoryDirectory(b'bar', 'bar', b'bar')
        t = directory_to_tree('', [child_ie], lambda p, x: None, {}, '.mydummy', allow_empty=False)
        self.assertTrue('.mydummy' in t)

    def test_empty_root(self):
        child_ie = InventoryDirectory(b'bar', 'bar', b'bar')
        t = directory_to_tree('', [child_ie], lambda p, x: None, {}, None, allow_empty=True)
        self.assertEqual(Tree(), t)

    def test_with_file(self):
        child_ie = InventoryFile(b'bar', 'bar', b'bar')
        b = Blob.from_string(b'bla')
        t1 = directory_to_tree('', [child_ie], lambda p, x: b.id, {}, None, allow_empty=False)
        t2 = Tree()
        t2.add(b'bar', 33188, b.id)
        self.assertEqual(t1, t2)

    def test_with_gitdir(self):
        child_ie = InventoryFile(b'bar', 'bar', b'bar')
        git_file_ie = InventoryFile(b'gitid', '.git', b'bar')
        b = Blob.from_string(b'bla')
        t1 = directory_to_tree('', [child_ie, git_file_ie], lambda p, x: b.id, {}, None, allow_empty=False)
        t2 = Tree()
        t2.add(b'bar', 33188, b.id)
        self.assertEqual(t1, t2)