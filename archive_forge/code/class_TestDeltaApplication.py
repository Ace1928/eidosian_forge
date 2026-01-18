from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
class TestDeltaApplication(TestCaseWithTransport):
    scenarios = delta_application_scenarios()

    def get_empty_inventory(self, reference_inv=None):
        """Get an empty inventory.

        Note that tests should not depend on the revision of the root for
        setting up test conditions, as it has to be flexible to accomodate non
        rich root repositories.

        :param reference_inv: If not None, get the revision for the root from
            this inventory. This is useful for dealing with older repositories
            that routinely discarded the root entry data. If None, the root's
            revision is set to 'basis'.
        """
        inv = inventory.Inventory()
        if reference_inv is not None:
            inv.root.revision = reference_inv.root.revision
        else:
            inv.root.revision = b'basis'
        return inv

    def make_file_ie(self, file_id=b'file-id', name='name', parent_id=None):
        ie_file = inventory.InventoryFile(file_id, name, parent_id)
        ie_file.revision = b'result'
        ie_file.text_size = 0
        ie_file.text_sha1 = b''
        return ie_file

    def test_empty_delta(self):
        inv = self.get_empty_inventory()
        delta = []
        inv = self.apply_delta(self, inv, delta)
        inv2 = self.get_empty_inventory(inv)
        self.assertEqual([], inv2._make_delta(inv))

    def test_None_file_id(self):
        inv = self.get_empty_inventory()
        dir1 = inventory.InventoryDirectory(b'dirid', 'dir1', inv.root.file_id)
        dir1.file_id = None
        dir1.revision = b'result'
        delta = [(None, 'dir1', None, dir1)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_unicode_file_id(self):
        inv = self.get_empty_inventory()
        dir1 = inventory.InventoryDirectory(b'dirid', 'dir1', inv.root.file_id)
        dir1.file_id = 'dirid'
        dir1.revision = b'result'
        delta = [(None, 'dir1', dir1.file_id, dir1)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_repeated_file_id(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'id', 'path1', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        file2 = file1.copy()
        file2.name = 'path2'
        delta = [(None, 'path1', b'id', file1), (None, 'path2', b'id', file2)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_repeated_new_path(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        file2 = file1.copy()
        file2.file_id = b'id2'
        delta = [(None, 'path', b'id1', file1), (None, 'path', b'id2', file2)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_repeated_old_path(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        file2 = inventory.InventoryFile(b'id2', 'path2', inv.root.file_id)
        file2.revision = b'result'
        file2.text_size = 0
        file2.text_sha1 = b''
        inv.add(file1)
        inv.add(file2)
        delta = [('path', None, b'id1', None), ('path', None, b'id2', None)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_mismatched_id_entry_id(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        delta = [(None, 'path', b'id', file1)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_mismatched_new_path_entry_None(self):
        inv = self.get_empty_inventory()
        delta = [(None, 'path', b'id', None)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_mismatched_new_path_None_entry(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        delta = [('path', None, b'id1', file1)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_parent_is_not_directory(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        file2 = inventory.InventoryFile(b'id2', 'path2', b'id1')
        file2.revision = b'result'
        file2.text_size = 0
        file2.text_sha1 = b''
        inv.add(file1)
        delta = [(None, 'path/path2', b'id2', file2)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_parent_is_missing(self):
        inv = self.get_empty_inventory()
        file2 = inventory.InventoryFile(b'id2', 'path2', b'missingparent')
        file2.revision = b'result'
        file2.text_size = 0
        file2.text_sha1 = b''
        delta = [(None, 'path/path2', b'id2', file2)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_new_parent_path_has_wrong_id(self):
        inv = self.get_empty_inventory()
        parent1 = inventory.InventoryDirectory(b'p-1', 'dir', inv.root.file_id)
        parent1.revision = b'result'
        parent2 = inventory.InventoryDirectory(b'p-2', 'dir2', inv.root.file_id)
        parent2.revision = b'result'
        file1 = inventory.InventoryFile(b'id', 'path', b'p-2')
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        inv.add(parent1)
        inv.add(parent2)
        delta = [(None, 'dir/path', b'id', file1)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_old_parent_path_is_wrong(self):
        inv = self.get_empty_inventory()
        parent1 = inventory.InventoryDirectory(b'p-1', 'dir', inv.root.file_id)
        parent1.revision = b'result'
        parent2 = inventory.InventoryDirectory(b'p-2', 'dir2', inv.root.file_id)
        parent2.revision = b'result'
        file1 = inventory.InventoryFile(b'id', 'path', b'p-2')
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        inv.add(parent1)
        inv.add(parent2)
        inv.add(file1)
        delta = [('dir/path', None, b'id', None)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_old_parent_path_is_for_other_id(self):
        inv = self.get_empty_inventory()
        parent1 = inventory.InventoryDirectory(b'p-1', 'dir', inv.root.file_id)
        parent1.revision = b'result'
        parent2 = inventory.InventoryDirectory(b'p-2', 'dir2', inv.root.file_id)
        parent2.revision = b'result'
        file1 = inventory.InventoryFile(b'id', 'path', b'p-2')
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        file2 = inventory.InventoryFile(b'id2', 'path', b'p-1')
        file2.revision = b'result'
        file2.text_size = 0
        file2.text_sha1 = b''
        inv.add(parent1)
        inv.add(parent2)
        inv.add(file1)
        inv.add(file2)
        delta = [('dir/path', None, b'id', None)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_add_existing_id_new_path(self):
        inv = self.get_empty_inventory()
        parent1 = inventory.InventoryDirectory(b'p-1', 'dir1', inv.root.file_id)
        parent1.revision = b'result'
        parent2 = inventory.InventoryDirectory(b'p-1', 'dir2', inv.root.file_id)
        parent2.revision = b'result'
        inv.add(parent1)
        delta = [(None, 'dir2', b'p-1', parent2)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_add_new_id_existing_path(self):
        inv = self.get_empty_inventory()
        parent1 = inventory.InventoryDirectory(b'p-1', 'dir1', inv.root.file_id)
        parent1.revision = b'result'
        parent2 = inventory.InventoryDirectory(b'p-2', 'dir1', inv.root.file_id)
        parent2.revision = b'result'
        inv.add(parent1)
        delta = [(None, 'dir1', b'p-2', parent2)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_remove_dir_leaving_dangling_child(self):
        inv = self.get_empty_inventory()
        dir1 = inventory.InventoryDirectory(b'p-1', 'dir1', inv.root.file_id)
        dir1.revision = b'result'
        dir2 = inventory.InventoryDirectory(b'p-2', 'child1', b'p-1')
        dir2.revision = b'result'
        dir3 = inventory.InventoryDirectory(b'p-3', 'child2', b'p-1')
        dir3.revision = b'result'
        inv.add(dir1)
        inv.add(dir2)
        inv.add(dir3)
        delta = [('dir1', None, b'p-1', None), ('dir1/child2', None, b'p-3', None)]
        self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)

    def test_add_file(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'file-id', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        delta = [(None, 'path', b'file-id', file1)]
        res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
        self.assertEqual(b'file-id', res_inv.get_entry(b'file-id').file_id)

    def test_remove_file(self):
        inv = self.get_empty_inventory()
        file1 = inventory.InventoryFile(b'file-id', 'path', inv.root.file_id)
        file1.revision = b'result'
        file1.text_size = 0
        file1.text_sha1 = b''
        inv.add(file1)
        delta = [('path', None, b'file-id', None)]
        res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
        self.assertEqual(None, res_inv.path2id('path'))
        self.assertRaises(errors.NoSuchId, res_inv.id2path, b'file-id')

    def test_rename_file(self):
        inv = self.get_empty_inventory()
        file1 = self.make_file_ie(name='path', parent_id=inv.root.file_id)
        inv.add(file1)
        file2 = self.make_file_ie(name='path2', parent_id=inv.root.file_id)
        delta = [('path', 'path2', b'file-id', file2)]
        res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
        self.assertEqual(None, res_inv.path2id('path'))
        self.assertEqual(b'file-id', res_inv.path2id('path2'))

    def test_replaced_at_new_path(self):
        inv = self.get_empty_inventory()
        file1 = self.make_file_ie(file_id=b'id1', parent_id=inv.root.file_id)
        inv.add(file1)
        file2 = self.make_file_ie(file_id=b'id2', parent_id=inv.root.file_id)
        delta = [('name', None, b'id1', None), (None, 'name', b'id2', file2)]
        res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
        self.assertEqual(b'id2', res_inv.path2id('name'))

    def test_rename_dir(self):
        inv = self.get_empty_inventory()
        dir1 = inventory.InventoryDirectory(b'dir-id', 'dir1', inv.root.file_id)
        dir1.revision = b'basis'
        file1 = self.make_file_ie(parent_id=b'dir-id')
        inv.add(dir1)
        inv.add(file1)
        dir2 = inventory.InventoryDirectory(b'dir-id', 'dir2', inv.root.file_id)
        dir2.revision = b'result'
        delta = [('dir1', 'dir2', b'dir-id', dir2)]
        res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
        self.assertEqual(b'file-id', res_inv.path2id('dir2/name'))

    def test_renamed_dir_with_renamed_child(self):
        inv = self.get_empty_inventory()
        dir1 = inventory.InventoryDirectory(b'dir-id', 'dir1', inv.root.file_id)
        dir1.revision = b'basis'
        file1 = self.make_file_ie(b'file-id-1', 'name1', parent_id=b'dir-id')
        file2 = self.make_file_ie(b'file-id-2', 'name2', parent_id=b'dir-id')
        inv.add(dir1)
        inv.add(file1)
        inv.add(file2)
        dir2 = inventory.InventoryDirectory(b'dir-id', 'dir2', inv.root.file_id)
        dir2.revision = b'result'
        file2b = self.make_file_ie(b'file-id-2', 'name2', inv.root.file_id)
        delta = [('dir1', 'dir2', b'dir-id', dir2), ('dir1/name2', 'name2', b'file-id-2', file2b)]
        res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
        self.assertEqual(b'file-id-1', res_inv.path2id('dir2/name1'))
        self.assertEqual(None, res_inv.path2id('dir2/name2'))
        self.assertEqual(b'file-id-2', res_inv.path2id('name2'))

    def test_is_root(self):
        """Ensure our root-checking code is accurate."""
        inv = inventory.Inventory(b'TREE_ROOT')
        self.assertTrue(inv.is_root(b'TREE_ROOT'))
        self.assertFalse(inv.is_root(b'booga'))
        inv.root.file_id = b'booga'
        self.assertFalse(inv.is_root(b'TREE_ROOT'))
        self.assertTrue(inv.is_root(b'booga'))
        inv.root = None
        self.assertFalse(inv.is_root(b'TREE_ROOT'))
        self.assertFalse(inv.is_root(b'booga'))

    def test_entries_for_empty_inventory(self):
        """Test that entries() will not fail for an empty inventory"""
        inv = Inventory(root_id=None)
        self.assertEqual([], inv.entries())