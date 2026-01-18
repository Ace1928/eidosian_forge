import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def _pending_merges(self):
    self.make_local_branch_and_tree()
    other_bzrdir = self.tree.controldir.sprout('other')
    other_tree = other_bzrdir.open_workingtree()
    self.build_tree_contents([('other/other-file', b'other')])
    other_tree.add('other-file')
    other_tree.commit('other commit', rev_id=b'other')
    self.tree.merge_from_branch(other_tree.branch)
    self.tree.revert(filenames=['other-file'], backups=False)