import os
from io import BytesIO
from ... import errors
from ... import revision as _mod_revision
from ...bzr.inventory import (Inventory, InventoryDirectory, InventoryFile,
from ...bzr.inventorytree import InventoryRevisionTree, InventoryTree
from ...tests import TestNotApplicable
from ...uncommit import uncommit
from .. import features
from ..per_workingtree import TestCaseWithWorkingTree
class UpdateToOneParentViaDeltaTests(TestCaseWithWorkingTree):
    """Tests for the update_basis_by_delta call.

    This is intuitively defined as 'apply an inventory delta to the basis and
    discard other parents', but for trees that have an inventory that is not
    managed as a tree-by-id, the implementation requires roughly duplicated
    tests with those for apply_inventory_delta on the main tree.
    """

    def assertDeltaApplicationResultsInExpectedBasis(self, tree, revid, delta, expected_inventory):
        with tree.lock_write():
            tree.update_basis_by_delta(revid, delta)
        self.assertEqual(revid, tree.last_revision())
        self.assertEqual([revid], tree.get_parent_ids())
        result_basis = tree.basis_tree()
        with result_basis.lock_read():
            self.assertEqual(expected_inventory, result_basis.root_inventory)

    def make_inv_delta(self, old, new):
        """Make an inventory delta from two inventories."""
        old_ids = set(old._byid)
        new_ids = set(new._byid)
        adds = new_ids - old_ids
        deletes = old_ids - new_ids
        common = old_ids.intersection(new_ids)
        delta = []
        for file_id in deletes:
            delta.append((old.id2path(file_id), None, file_id, None))
        for file_id in adds:
            delta.append((None, new.id2path(file_id), file_id, new.get_entry(file_id)))
        for file_id in common:
            if old.get_entry(file_id) != new.get_entry(file_id):
                delta.append((old.id2path(file_id), new.id2path(file_id), file_id, new.get_entry(file_id)))
        return delta

    def fake_up_revision(self, tree, revid, shape):
        if not isinstance(tree, InventoryTree):
            raise TestNotApplicable('test requires inventory tree')

        class ShapeTree(InventoryRevisionTree):

            def __init__(self, shape):
                self._repository = tree.branch.repository
                self._inventory = shape

            def get_file_text(self, path):
                file_id = self.path2id(path)
                ie = self.root_inventory.get_entry(file_id)
                if ie.kind != 'file':
                    return b''
                return b'a' * ie.text_size

            def get_file(self, path):
                return BytesIO(self.get_file_text(path))
        with tree.lock_write():
            if shape.root.revision is None:
                shape.root.revision = revid
            builder = tree.branch.get_commit_builder(parents=[], timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', revision_id=revid)
            shape_tree = ShapeTree(shape)
            base_tree = tree.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
            changes = shape_tree.iter_changes(base_tree)
            list(builder.record_iter_changes(shape_tree, base_tree.get_revision_id(), changes))
            builder.finish_inventory()
            builder.commit('Message')

    def add_entry(self, inv, rev_id, entry):
        entry.revision = rev_id
        inv.add(entry)

    def add_dir(self, inv, rev_id, file_id, parent_id, name):
        new_dir = InventoryDirectory(file_id, name, parent_id)
        self.add_entry(inv, rev_id, new_dir)

    def add_file(self, inv, rev_id, file_id, parent_id, name, sha, size):
        new_file = InventoryFile(file_id, name, parent_id)
        new_file.text_sha1 = sha
        new_file.text_size = size
        self.add_entry(inv, rev_id, new_file)

    def add_link(self, inv, rev_id, file_id, parent_id, name, target):
        new_link = InventoryLink(file_id, name, parent_id)
        new_link.symlink_target = target
        self.add_entry(inv, rev_id, new_link)

    def add_new_root(self, new_shape, old_revid, new_revid):
        if self.bzrdir_format.repository_format.rich_root_data:
            self.add_dir(new_shape, old_revid, b'root-id', None, '')
        else:
            self.add_dir(new_shape, new_revid, b'root-id', None, '')

    def assertTransitionFromBasisToShape(self, basis_shape, basis_revid, new_shape, new_revid, extra_parent=None, set_current_inventory=True):
        basis_shape.revision_id = basis_revid
        new_shape.revision_id = new_revid
        delta = self.make_inv_delta(basis_shape, new_shape)
        tree = self.make_branch_and_tree('tree')
        if basis_revid is not None:
            self.fake_up_revision(tree, basis_revid, basis_shape)
            parents = [basis_revid]
            if extra_parent is not None:
                parents.append(extra_parent)
            tree.set_parent_ids(parents)
        self.fake_up_revision(tree, new_revid, new_shape)
        if set_current_inventory:
            tree._write_inventory(new_shape)
        self.assertDeltaApplicationResultsInExpectedBasis(tree, new_revid, delta, new_shape)
        tree._validate()
        if tree.user_url != tree.branch.user_url:
            tree.branch.controldir.root_transport.delete_tree('.')
        tree.controldir.root_transport.delete_tree('.')

    def test_no_parents_just_root(self):
        """Test doing an empty commit - no parent, set a root only."""
        basis_shape = Inventory(root_id=None)
        new_shape = Inventory()
        self.assertTransitionFromBasisToShape(basis_shape, None, new_shape, b'new_parent')

    def test_no_parents_full_tree(self):
        """Test doing a regular initial commit with files and dirs."""
        basis_shape = Inventory(root_id=None)
        revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_dir(new_shape, revid, b'root-id', None, '')
        self.add_link(new_shape, revid, b'link-id', b'root-id', 'link', 'target')
        self.add_file(new_shape, revid, b'file-id', b'root-id', 'file', b'1' * 32, 12)
        self.add_dir(new_shape, revid, b'dir-id', b'root-id', 'dir')
        self.add_file(new_shape, revid, b'subfile-id', b'dir-id', 'subfile', b'2' * 32, 24)
        self.assertTransitionFromBasisToShape(basis_shape, None, new_shape, revid)

    def test_file_content_change(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_file(basis_shape, old_revid, b'file-id', b'root-id', 'file', b'1' * 32, 12)
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_file(new_shape, new_revid, b'file-id', b'root-id', 'file', b'2' * 32, 24)
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_link_content_change(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_link(basis_shape, old_revid, b'link-id', b'root-id', 'link', 'old-target')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_link(new_shape, new_revid, b'link-id', b'root-id', 'link', 'new-target')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_kind_changes(self):

        def do_file(inv, revid):
            self.add_file(inv, revid, b'path-id', b'root-id', 'path', b'1' * 32, 12)

        def do_link(inv, revid):
            self.add_link(inv, revid, b'path-id', b'root-id', 'path', 'target')

        def do_dir(inv, revid):
            self.add_dir(inv, revid, b'path-id', b'root-id', 'path')
        for old_factory in (do_file, do_link, do_dir):
            for new_factory in (do_file, do_link, do_dir):
                if old_factory == new_factory:
                    continue
                old_revid = b'old-parent'
                basis_shape = Inventory(root_id=None)
                self.add_dir(basis_shape, old_revid, b'root-id', None, '')
                old_factory(basis_shape, old_revid)
                new_revid = b'new-parent'
                new_shape = Inventory(root_id=None)
                self.add_new_root(new_shape, old_revid, new_revid)
                new_factory(new_shape, new_revid)
                self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_content_from_second_parent_is_dropped(self):
        left_revid = b'left-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, left_revid, b'root-id', None, '')
        self.add_link(basis_shape, left_revid, b'link-id', b'root-id', 'link', 'left-target')
        right_revid = b'right-parent'
        right_shape = Inventory(root_id=None)
        self.add_dir(right_shape, left_revid, b'root-id', None, '')
        self.add_link(right_shape, right_revid, b'link-id', b'root-id', 'link', 'some-target')
        self.add_dir(right_shape, right_revid, b'subdir-id', b'root-id', 'dir')
        self.add_file(right_shape, right_revid, b'file-id', b'subdir-id', 'file', b'2' * 32, 24)
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, left_revid, new_revid)
        self.add_link(new_shape, new_revid, b'link-id', b'root-id', 'link', 'new-target')
        self.assertTransitionFromBasisToShape(basis_shape, left_revid, new_shape, new_revid, right_revid)

    def test_parent_id_changed(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'orig-parent-id', b'root-id', 'dir')
        self.add_dir(basis_shape, old_revid, b'dir-id', b'orig-parent-id', 'dir')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'new-parent-id', b'root-id', 'dir')
        self.add_dir(new_shape, new_revid, b'dir-id', b'new-parent-id', 'dir')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_name_changed(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'parent-id', b'root-id', 'origdir')
        self.add_dir(basis_shape, old_revid, b'dir-id', b'parent-id', 'olddir')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'parent-id', b'root-id', 'newdir')
        self.add_dir(new_shape, new_revid, b'dir-id', b'parent-id', 'newdir')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_parent_child_swap(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_dir(basis_shape, old_revid, b'dir-id-B', b'dir-id-A', 'B')
        self.add_link(basis_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'C')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'dir-id-B', b'root-id', 'A')
        self.add_dir(new_shape, new_revid, b'dir-id-A', b'dir-id-B', 'B')
        self.add_link(new_shape, new_revid, b'link-id-C', b'dir-id-A', 'C', 'C')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_parent_deleted_child_renamed(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_dir(basis_shape, old_revid, b'dir-id-B', b'dir-id-A', 'B')
        self.add_link(basis_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'C')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'dir-id-B', b'root-id', 'A')
        self.add_link(new_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'C')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_dir_to_root(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_link(basis_shape, old_revid, b'link-id-B', b'dir-id-A', 'B', 'B')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_dir(new_shape, new_revid, b'dir-id-A', None, '')
        self.add_link(new_shape, old_revid, b'link-id-B', b'dir-id-A', 'B', 'B')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_path_swap(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_dir(basis_shape, old_revid, b'dir-id-B', b'root-id', 'B')
        self.add_link(basis_shape, old_revid, b'link-id-C', b'root-id', 'C', 'C')
        self.add_link(basis_shape, old_revid, b'link-id-D', b'root-id', 'D', 'D')
        self.add_file(basis_shape, old_revid, b'file-id-E', b'root-id', 'E', b'1' * 32, 12)
        self.add_file(basis_shape, old_revid, b'file-id-F', b'root-id', 'F', b'2' * 32, 24)
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'dir-id-A', b'root-id', 'B')
        self.add_dir(new_shape, new_revid, b'dir-id-B', b'root-id', 'A')
        self.add_link(new_shape, new_revid, b'link-id-C', b'root-id', 'D', 'C')
        self.add_link(new_shape, new_revid, b'link-id-D', b'root-id', 'C', 'D')
        self.add_file(new_shape, new_revid, b'file-id-E', b'root-id', 'F', b'1' * 32, 12)
        self.add_file(new_shape, new_revid, b'file-id-F', b'root-id', 'E', b'2' * 32, 24)
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_adds(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'dir-id-A', b'root-id', 'A')
        self.add_link(new_shape, new_revid, b'link-id-B', b'root-id', 'B', 'C')
        self.add_file(new_shape, new_revid, b'file-id-C', b'root-id', 'C', b'1' * 32, 12)
        self.add_file(new_shape, new_revid, b'file-id-D', b'dir-id-A', 'D', b'2' * 32, 24)
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_removes(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_link(basis_shape, old_revid, b'link-id-B', b'root-id', 'B', 'C')
        self.add_file(basis_shape, old_revid, b'file-id-C', b'root-id', 'C', b'1' * 32, 12)
        self.add_file(basis_shape, old_revid, b'file-id-D', b'dir-id-A', 'D', b'2' * 32, 24)
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_move_to_added_dir(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_link(basis_shape, old_revid, b'link-id-B', b'root-id', 'B', 'C')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'dir-id-A', b'root-id', 'A')
        self.add_link(new_shape, new_revid, b'link-id-B', b'dir-id-A', 'B', 'C')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_move_from_removed_dir(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_link(basis_shape, old_revid, b'link-id-B', b'dir-id-A', 'B', 'C')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_link(new_shape, new_revid, b'link-id-B', b'root-id', 'B', 'C')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_move_moves_children_recursively(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_dir(basis_shape, old_revid, b'dir-id-B', b'dir-id-A', 'B')
        self.add_link(basis_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'D')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, new_revid, b'dir-id-A', b'root-id', 'B')
        self.add_dir(new_shape, old_revid, b'dir-id-B', b'dir-id-A', 'B')
        self.add_link(new_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'D')
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)

    def test_add_files_to_empty_directory(self):
        old_revid = b'old-parent'
        basis_shape = Inventory(root_id=None)
        self.add_dir(basis_shape, old_revid, b'root-id', None, '')
        self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        new_revid = b'new-parent'
        new_shape = Inventory(root_id=None)
        self.add_new_root(new_shape, old_revid, new_revid)
        self.add_dir(new_shape, old_revid, b'dir-id-A', b'root-id', 'A')
        self.add_file(new_shape, new_revid, b'file-id-B', b'dir-id-A', 'B', b'1' * 32, 24)
        self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid, set_current_inventory=False)