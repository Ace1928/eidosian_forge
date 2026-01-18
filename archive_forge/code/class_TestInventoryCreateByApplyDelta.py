from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
class TestInventoryCreateByApplyDelta(TestInventory):
    """A subset of the inventory delta application tests.

    See test_inv which has comprehensive delta application tests for
    inventories, dirstate, and repository based inventories.
    """

    def test_add(self):
        inv = self.make_init_inventory()
        inv = inv.create_by_apply_delta([(None, 'a', b'a-id', self.make_file(b'a-id', 'a', b'tree-root'))], b'new-test-rev')
        self.assertEqual('a', inv.id2path(b'a-id'))

    def test_delete(self):
        inv = self.make_init_inventory()
        inv = inv.create_by_apply_delta([(None, 'a', b'a-id', self.make_file(b'a-id', 'a', b'tree-root'))], b'new-rev-1')
        self.assertEqual('a', inv.id2path(b'a-id'))
        inv = inv.create_by_apply_delta([('a', None, b'a-id', None)], b'new-rev-2')
        self.assertRaises(errors.NoSuchId, inv.id2path, b'a-id')

    def test_rename(self):
        inv = self.make_init_inventory()
        inv = inv.create_by_apply_delta([(None, 'a', b'a-id', self.make_file(b'a-id', 'a', b'tree-root'))], b'new-rev-1')
        self.assertEqual('a', inv.id2path(b'a-id'))
        a_ie = inv.get_entry(b'a-id')
        b_ie = self.make_file(a_ie.file_id, 'b', a_ie.parent_id)
        inv = inv.create_by_apply_delta([('a', 'b', b'a-id', b_ie)], b'new-rev-2')
        self.assertEqual('b', inv.id2path(b'a-id'))

    def test_illegal(self):
        inv = self.make_init_inventory()
        self.assertRaises(errors.InconsistentDelta, inv.create_by_apply_delta, [(None, 'a', b'id-1', self.make_file(b'id-1', 'a', b'tree-root')), (None, 'b', b'id-1', self.make_file(b'id-1', 'b', b'tree-root'))], b'new-rev-1')