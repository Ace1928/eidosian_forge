import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
class TestInventoryAltered(TestCaseWithTransport):

    def test_inventory_altered_unchanged(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo', ids=b'foo-id')
        with tree.preview_transform() as tt:
            self.assertEqual([], tt._inventory_altered())

    def test_inventory_altered_changed_parent_id(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo', ids=b'foo-id')
        with tree.preview_transform() as tt:
            tt.unversion_file(tt.root)
            tt.version_file(tt.root, file_id=b'new-id')
            foo_trans_id = tt.trans_id_tree_path('foo')
            foo_tuple = ('foo', foo_trans_id)
            root_tuple = ('', tt.root)
            self.assertEqual([root_tuple, foo_tuple], tt._inventory_altered())

    def test_inventory_altered_noop_changed_parent_id(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo', ids=b'foo-id')
        with tree.preview_transform() as tt:
            tt.unversion_file(tt.root)
            tt.version_file(tt.root, file_id=tree.path2id(''))
            tt.trans_id_tree_path('foo')
            self.assertEqual([], tt._inventory_altered())