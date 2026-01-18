import os
from breezy import branch, controldir, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import bzrdir
from breezy.bzr.knitrepo import RepositoryFormatKnit1
from breezy.tests import fixtures, test_server
from breezy.tests.blackbox import test_switch
from breezy.tests.features import HardlinkFeature
from breezy.tests.script import run_script
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.urlutils import local_path_to_url, strip_trailing_slash
from breezy.workingtree import WorkingTree
def _checkout_and_branch(self, option=''):
    self.script_runner.run_script(self, '\n                $ brz checkout %(option)s repo/trunk checkout\n                $ cd checkout\n                $ brz branch --switch ../repo/trunk ../repo/branched\n                2>Branched 0 revisions.\n                2>Tree is up to date at revision 0.\n                2>Switched to branch:...branched...\n                $ cd ..\n                ' % locals())
    bound_branch = branch.Branch.open_containing('checkout')[0]
    master_branch = branch.Branch.open_containing('repo/branched')[0]
    return (bound_branch, master_branch)