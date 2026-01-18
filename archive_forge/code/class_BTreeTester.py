import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
class BTreeTester(tests.TestCase):
    """A simple unittest tester for the BundleTree class."""

    def make_tree_1(self):
        mtree = MockTree()
        mtree.add_dir(b'a', 'grandparent')
        mtree.add_dir(b'b', 'grandparent/parent')
        mtree.add_file(b'c', 'grandparent/parent/file', b'Hello\n')
        mtree.add_dir(b'd', 'grandparent/alt_parent')
        return (BundleTree(mtree, b''), mtree)

    def test_renames(self):
        """Ensure that file renames have the proper effect on children"""
        btree = self.make_tree_1()[0]
        self.assertEqual(btree.old_path('grandparent'), 'grandparent')
        self.assertEqual(btree.old_path('grandparent/parent'), 'grandparent/parent')
        self.assertEqual(btree.old_path('grandparent/parent/file'), 'grandparent/parent/file')
        self.assertEqual(btree.id2path(b'a'), 'grandparent')
        self.assertEqual(btree.id2path(b'b'), 'grandparent/parent')
        self.assertEqual(btree.id2path(b'c'), 'grandparent/parent/file')
        self.assertEqual(btree.path2id('grandparent'), b'a')
        self.assertEqual(btree.path2id('grandparent/parent'), b'b')
        self.assertEqual(btree.path2id('grandparent/parent/file'), b'c')
        self.assertIs(btree.path2id('grandparent2'), None)
        self.assertIs(btree.path2id('grandparent2/parent'), None)
        self.assertIs(btree.path2id('grandparent2/parent/file'), None)
        btree.note_rename('grandparent', 'grandparent2')
        self.assertIs(btree.old_path('grandparent'), None)
        self.assertIs(btree.old_path('grandparent/parent'), None)
        self.assertIs(btree.old_path('grandparent/parent/file'), None)
        self.assertEqual(btree.id2path(b'a'), 'grandparent2')
        self.assertEqual(btree.id2path(b'b'), 'grandparent2/parent')
        self.assertEqual(btree.id2path(b'c'), 'grandparent2/parent/file')
        self.assertEqual(btree.path2id('grandparent2'), b'a')
        self.assertEqual(btree.path2id('grandparent2/parent'), b'b')
        self.assertEqual(btree.path2id('grandparent2/parent/file'), b'c')
        self.assertTrue(btree.path2id('grandparent') is None)
        self.assertTrue(btree.path2id('grandparent/parent') is None)
        self.assertTrue(btree.path2id('grandparent/parent/file') is None)
        btree.note_rename('grandparent/parent', 'grandparent2/parent2')
        self.assertEqual(btree.id2path(b'a'), 'grandparent2')
        self.assertEqual(btree.id2path(b'b'), 'grandparent2/parent2')
        self.assertEqual(btree.id2path(b'c'), 'grandparent2/parent2/file')
        self.assertEqual(btree.path2id('grandparent2'), b'a')
        self.assertEqual(btree.path2id('grandparent2/parent2'), b'b')
        self.assertEqual(btree.path2id('grandparent2/parent2/file'), b'c')
        self.assertTrue(btree.path2id('grandparent2/parent') is None)
        self.assertTrue(btree.path2id('grandparent2/parent/file') is None)
        btree.note_rename('grandparent/parent/file', 'grandparent2/parent2/file2')
        self.assertEqual(btree.id2path(b'a'), 'grandparent2')
        self.assertEqual(btree.id2path(b'b'), 'grandparent2/parent2')
        self.assertEqual(btree.id2path(b'c'), 'grandparent2/parent2/file2')
        self.assertEqual(btree.path2id('grandparent2'), b'a')
        self.assertEqual(btree.path2id('grandparent2/parent2'), b'b')
        self.assertEqual(btree.path2id('grandparent2/parent2/file2'), b'c')
        self.assertTrue(btree.path2id('grandparent2/parent2/file') is None)

    def test_moves(self):
        """Ensure that file moves have the proper effect on children"""
        btree = self.make_tree_1()[0]
        btree.note_rename('grandparent/parent/file', 'grandparent/alt_parent/file')
        self.assertEqual(btree.id2path(b'c'), 'grandparent/alt_parent/file')
        self.assertEqual(btree.path2id('grandparent/alt_parent/file'), b'c')
        self.assertTrue(btree.path2id('grandparent/parent/file') is None)

    def unified_diff(self, old, new):
        out = BytesIO()
        diff.internal_diff('old', old, 'new', new, out)
        out.seek(0, 0)
        return out.read()

    def make_tree_2(self):
        btree = self.make_tree_1()[0]
        btree.note_rename('grandparent/parent/file', 'grandparent/alt_parent/file')
        self.assertRaises(errors.NoSuchId, btree.id2path, b'e')
        self.assertFalse(btree.is_versioned('grandparent/parent/file'))
        btree.note_id(b'e', 'grandparent/parent/file')
        return btree

    def test_adds(self):
        """File/inventory adds"""
        btree = self.make_tree_2()
        add_patch = self.unified_diff([], [b'Extra cheese\n'])
        btree.note_patch('grandparent/parent/file', add_patch)
        btree.note_id(b'f', 'grandparent/parent/symlink', kind='symlink')
        btree.note_target('grandparent/parent/symlink', 'venus')
        self.adds_test(btree)

    def adds_test(self, btree):
        self.assertEqual(btree.id2path(b'e'), 'grandparent/parent/file')
        self.assertEqual(btree.path2id('grandparent/parent/file'), b'e')
        with btree.get_file('grandparent/parent/file') as f:
            self.assertEqual(f.read(), b'Extra cheese\n')
        self.assertEqual(btree.get_symlink_target('grandparent/parent/symlink'), 'venus')

    def make_tree_3(self):
        btree, mtree = self.make_tree_1()
        mtree.add_file(b'e', 'grandparent/parent/topping', b'Anchovies\n')
        btree.note_rename('grandparent/parent/file', 'grandparent/alt_parent/file')
        btree.note_rename('grandparent/parent/topping', 'grandparent/alt_parent/stopping')
        return btree

    def get_file_test(self, btree):
        with btree.get_file(btree.id2path(b'e')) as f:
            self.assertEqual(f.read(), b'Lemon\n')
        with btree.get_file(btree.id2path(b'c')) as f:
            self.assertEqual(f.read(), b'Hello\n')

    def test_get_file(self):
        """Get file contents"""
        btree = self.make_tree_3()
        mod_patch = self.unified_diff([b'Anchovies\n'], [b'Lemon\n'])
        btree.note_patch('grandparent/alt_parent/stopping', mod_patch)
        self.get_file_test(btree)

    def test_delete(self):
        """Deletion by bundle"""
        btree = self.make_tree_1()[0]
        with btree.get_file(btree.id2path(b'c')) as f:
            self.assertEqual(f.read(), b'Hello\n')
        btree.note_deletion('grandparent/parent/file')
        self.assertRaises(errors.NoSuchId, btree.id2path, b'c')
        self.assertFalse(btree.is_versioned('grandparent/parent/file'))

    def sorted_ids(self, tree):
        ids = sorted(tree.all_file_ids())
        return ids

    def test_iteration(self):
        """Ensure that iteration through ids works properly"""
        btree = self.make_tree_1()[0]
        self.assertEqual(self.sorted_ids(btree), [inventory.ROOT_ID, b'a', b'b', b'c', b'd'])
        btree.note_deletion('grandparent/parent/file')
        btree.note_id(b'e', 'grandparent/alt_parent/fool', kind='directory')
        btree.note_last_changed('grandparent/alt_parent/fool', 'revisionidiguess')
        self.assertEqual(self.sorted_ids(btree), [inventory.ROOT_ID, b'a', b'b', b'd', b'e'])