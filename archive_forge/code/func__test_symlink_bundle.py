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
def _test_symlink_bundle(self, link_name, link_target, new_link_target):
    link_id = b'link-1'
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    tt = self.tree1.transform()
    tt.new_symlink(link_name, tt.root, link_target, link_id)
    tt.apply()
    self.tree1.commit('add symlink', rev_id=b'l@cset-0-1')
    bundle = self.get_valid_bundle(b'null:', b'l@cset-0-1')
    if getattr(bundle, 'revision_tree', None) is not None:
        bund_tree = bundle.revision_tree(self.b1.repository, b'l@cset-0-1')
        self.assertEqual(link_target, bund_tree.get_symlink_target(link_name))
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path(link_name)
    tt.adjust_path('link2', tt.root, trans_id)
    tt.delete_contents(trans_id)
    tt.create_symlink(new_link_target, trans_id)
    tt.apply()
    self.tree1.commit('rename and change symlink', rev_id=b'l@cset-0-2')
    bundle = self.get_valid_bundle(b'l@cset-0-1', b'l@cset-0-2')
    if getattr(bundle, 'revision_tree', None) is not None:
        bund_tree = bundle.revision_tree(self.b1.repository, b'l@cset-0-2')
        self.assertEqual(new_link_target, bund_tree.get_symlink_target('link2'))
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path('link2')
    tt.delete_contents(trans_id)
    tt.create_symlink('jupiter', trans_id)
    tt.apply()
    self.tree1.commit('just change symlink target', rev_id=b'l@cset-0-3')
    bundle = self.get_valid_bundle(b'l@cset-0-2', b'l@cset-0-3')
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path('link2')
    tt.delete_contents(trans_id)
    tt.apply()
    self.tree1.commit('Delete symlink', rev_id=b'l@cset-0-4')
    bundle = self.get_valid_bundle(b'l@cset-0-3', b'l@cset-0-4')