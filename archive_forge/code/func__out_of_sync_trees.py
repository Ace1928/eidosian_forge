import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def _out_of_sync_trees(self):
    self.make_local_branch_and_tree()
    self.run_bzr(['checkout', '--lightweight', 'local', 'checkout'])
    self.build_tree_contents([('local/file', b'modified in local')])
    self.tree.commit('modify file', rev_id=b'modified-in-local')
    self._default_wd = 'checkout'
    self._default_errors = ["Working tree is out of date, please run 'brz update'\\."]