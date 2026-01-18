import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def _test_shelve_symlink_creation(self, link_name, link_target, shelve_change=False):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.commit('Empty tree')
    os.symlink(link_target, link_name)
    tree.add(link_name, ids=b'foo-id')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('add file', b'foo-id', 'symlink', link_name)], list(creator.iter_shelvable()))
    if shelve_change:
        creator.shelve_change(('add file', b'foo-id', 'symlink', link_name))
    else:
        creator.shelve_creation(b'foo-id')
    creator.transform()
    s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
    self.assertPathDoesNotExist(link_name)
    limbo_name = creator.shelf_transform._limbo_name(s_trans_id)
    self.assertEqual(link_target, osutils.readlink(limbo_name))
    ptree = creator.shelf_transform.get_preview_tree()
    self.assertEqual(link_target, ptree.get_symlink_target(ptree.id2path(b'foo-id')))