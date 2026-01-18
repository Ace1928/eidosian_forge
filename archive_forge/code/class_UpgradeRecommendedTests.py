import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
class UpgradeRecommendedTests(TestCaseWithTransport):

    def test_recommend_upgrade_wt4(self):
        self.run_bzr('init --format=knit a')
        out, err = self.run_bzr('status a')
        self.assertContainsRe(err, 'brz upgrade .*[/\\\\]a')

    def test_no_upgrade_recommendation_from_bzrdir(self):
        self.run_bzr('init --format=knit a')
        out, err = self.run_bzr('revno a')
        if err.find('upgrade') > -1:
            self.fail("message shouldn't suggest upgrade:\n%s" % err)

    def test_upgrade_shared_repo(self):
        repo = self.make_repository('repo', format='2a', shared=True)
        branch = self.make_branch_and_tree('repo/branch', format='pack-0.92')
        self.get_transport('repo/branch/.bzr/repository').delete_tree('.')
        out, err = self.run_bzr(['upgrade'], working_dir='repo/branch')