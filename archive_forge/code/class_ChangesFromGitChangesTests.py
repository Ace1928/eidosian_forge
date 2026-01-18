import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
class ChangesFromGitChangesTests(TestCase):

    def setUp(self):
        super().setUp()
        self.maxDiff = None
        self.mapping = default_mapping

    def transform(self, changes, specific_files=None, include_unchanged=False, source_extras=None, target_extras=None):
        return list(changes_from_git_changes(changes, self.mapping, specific_files=specific_files, include_unchanged=include_unchanged, source_extras=source_extras, target_extras=target_extras))

    def test_empty(self):
        self.assertEqual([], self.transform([]))

    def test_modified(self):
        a = Blob.from_string(b'a')
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(b'git:a', ('a', 'a'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'file'), (False, False), False)], self.transform([('modify', (b'a', stat.S_IFREG | 420, a), (b'a', stat.S_IFREG | 420, b))]))

    def test_kind_changed(self):
        a = Blob.from_string(b'a')
        b = Blob.from_string(b'target')
        self.assertEqual([TreeChange(b'git:a', ('a', 'a'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'symlink'), (False, False), False)], self.transform([('modify', (b'a', stat.S_IFREG | 420, a), (b'a', stat.S_IFLNK, b))]))

    def test_rename_no_changes(self):
        a = Blob.from_string(b'a')
        self.assertEqual([TreeChange(b'git:old', ('old', 'a'), False, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('old', 'a'), ('file', 'file'), (False, False), False)], self.transform([('rename', (b'old', stat.S_IFREG | 420, a), (b'a', stat.S_IFREG | 420, a))]))

    def test_rename_and_modify(self):
        a = Blob.from_string(b'a')
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(b'git:a', ('a', 'b'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'b'), ('file', 'file'), (False, False), False)], self.transform([('rename', (b'a', stat.S_IFREG | 420, a), (b'b', stat.S_IFREG | 420, b))]))

    def test_copy_no_changes(self):
        a = Blob.from_string(b'a')
        self.assertEqual([TreeChange(b'git:a', ('old', 'a'), False, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('old', 'a'), ('file', 'file'), (False, False), True)], self.transform([('copy', (b'old', stat.S_IFREG | 420, a), (b'a', stat.S_IFREG | 420, a))]))

    def test_copy_and_modify(self):
        a = Blob.from_string(b'a')
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(b'git:b', ('a', 'b'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'b'), ('file', 'file'), (False, False), True)], self.transform([('copy', (b'a', stat.S_IFREG | 420, a), (b'b', stat.S_IFREG | 420, b))]))

    def test_add(self):
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(b'git:a', (None, 'a'), True, (False, True), (None, b'TREE_ROOT'), (None, 'a'), (None, 'file'), (None, False), False)], self.transform([('add', (None, None, None), (b'a', stat.S_IFREG | 420, b))]))

    def test_delete(self):
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(b'git:a', ('a', None), True, (True, False), (b'TREE_ROOT', None), ('a', None), ('file', None), (False, None), False)], self.transform([('remove', (b'a', stat.S_IFREG | 420, b), (None, None, None))]))

    def test_unchanged(self):
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(b'git:a', ('a', 'a'), False, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'file'), (False, False), False)], self.transform([('unchanged', (b'a', stat.S_IFREG | 420, b), (b'a', stat.S_IFREG | 420, b))], include_unchanged=True))
        self.assertEqual([], self.transform([('unchanged', (b'a', stat.S_IFREG | 420, b), (b'a', stat.S_IFREG | 420, b))], include_unchanged=False))

    def test_unversioned(self):
        b = Blob.from_string(b'b')
        self.assertEqual([TreeChange(None, (None, 'a'), True, (False, False), (None, b'TREE_ROOT'), (None, 'a'), (None, 'file'), (None, False), False)], self.transform([('add', (None, None, None), (b'a', stat.S_IFREG | 420, b))], target_extras={b'a'}))
        self.assertEqual([TreeChange(None, ('a', 'a'), False, (False, False), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'file'), (False, False), False)], self.transform([('add', (b'a', stat.S_IFREG | 420, b), (b'a', stat.S_IFREG | 420, b))], source_extras={b'a'}, target_extras={b'a'}))