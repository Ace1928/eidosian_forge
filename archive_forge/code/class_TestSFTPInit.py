import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
class TestSFTPInit(TestCaseWithSFTPServer):

    def test_init(self):
        out, err = self.run_bzr(['init', '--format=pack-0.92', self.get_url()])
        self.assertEqual(out, 'Created a standalone branch (format: pack-0.92)\n')

    def test_init_existing_branch(self):
        self.make_branch('.')
        out, err = self.run_bzr_error(['Already a branch'], ['init', self.get_url()])
        self.assertFalse(re.search('use brz checkout', err))

    def test_init_existing_branch_with_workingtree(self):
        self.make_branch_and_tree('.')
        self.run_bzr_error(['Already a branch'], ['init', self.get_url()])

    def test_init_append_revisions_only(self):
        self.run_bzr('init --format=dirstate-tags normal_branch6')
        branch = _mod_branch.Branch.open('normal_branch6')
        self.assertEqual(None, branch.get_append_revisions_only())
        self.run_bzr('init --append-revisions-only --format=dirstate-tags branch6')
        branch = _mod_branch.Branch.open('branch6')
        self.assertEqual(True, branch.get_append_revisions_only())
        self.run_bzr_error(['cannot be set to append-revisions-only'], 'init --append-revisions-only --format=knit knit')

    def test_init_without_username(self):
        """Ensure init works if username is not set.
        """
        self.overrideEnv('EMAIL', None)
        self.overrideEnv('BRZ_EMAIL', None)
        out, err = self.run_bzr(['init', 'foo'])
        self.assertEqual(err, '')
        self.assertTrue(os.path.exists('foo'))