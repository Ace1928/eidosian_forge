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
def _valid_apply_bundle(self, base_rev_id, info, to_tree):
    original_parents = to_tree.get_parent_ids()
    repository = to_tree.branch.repository
    original_parents = to_tree.get_parent_ids()
    self.assertIs(repository.has_revision(base_rev_id), True)
    for rev in info.real_revisions:
        self.assertTrue(not repository.has_revision(rev.revision_id), 'Revision {%s} present before applying bundle' % rev.revision_id)
    merge_bundle(info, to_tree, True, merge.Merge3Merger, False, False)
    for rev in info.real_revisions:
        self.assertTrue(repository.has_revision(rev.revision_id), 'Missing revision {%s} after applying bundle' % rev.revision_id)
    self.assertTrue(to_tree.branch.repository.has_revision(info.target))
    self.assertEqual(original_parents + [info.target], to_tree.get_parent_ids())
    rev = info.real_revisions[-1]
    base_tree = self.b1.repository.revision_tree(rev.revision_id)
    to_tree = to_tree.branch.repository.revision_tree(rev.revision_id)
    base_files = list(base_tree.list_files())
    to_files = list(to_tree.list_files())
    self.assertEqual(len(base_files), len(to_files))
    for base_file, to_file in zip(base_files, to_files):
        self.assertEqual(base_file, to_file)
    for path, status, kind, entry in base_files:
        to_path = InterTree.get(base_tree, to_tree).find_target_path(path)
        self.assertEqual(base_tree.get_file_size(path), to_tree.get_file_size(to_path))
        self.assertEqual(base_tree.get_file_sha1(path), to_tree.get_file_sha1(to_path))