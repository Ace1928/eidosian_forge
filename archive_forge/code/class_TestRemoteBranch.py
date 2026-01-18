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
class TestRemoteBranch(TestCaseWithSFTPServer):

    def setUp(self):
        super().setUp()
        self.skipTest('tests often hang - see pad.lv/1997033')
        tree = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/file', b'file content\n')])
        tree.add('file')
        tree.commit('file created')

    def test_branch_local_remote(self):
        self.run_bzr(['branch', 'branch', self.get_url('remote')])
        t = self.get_transport()
        self.assertFalse(t.has('remote/file'))

    def test_branch_remote_remote(self):
        self.run_bzr(['branch', self.get_url('branch'), self.get_url('remote')])
        t = self.get_transport()
        self.assertFalse(t.has('remote/file'))