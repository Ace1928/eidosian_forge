from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
class TestSmartUpgrade(tests.TestCaseWithTransport):
    from_format = controldir.format_registry.make_controldir('pack-0.92')
    to_format = controldir.format_registry.make_controldir('2a')

    def make_standalone_branch(self):
        wt = self.make_branch_and_tree('branch1', format=self.from_format)
        return wt.controldir

    def test_upgrade_standalone_branch(self):
        control = self.make_standalone_branch()
        tried, worked, issues = upgrade.smart_upgrade([control], format=self.to_format)
        self.assertLength(1, tried)
        self.assertEqual(tried[0], control)
        self.assertLength(1, worked)
        self.assertEqual(worked[0], control)
        self.assertLength(0, issues)
        self.assertPathExists('branch1/backup.bzr.~1~')
        self.assertEqual(control.open_repository()._format, self.to_format._repository_format)

    def test_upgrade_standalone_branch_cleanup(self):
        control = self.make_standalone_branch()
        tried, worked, issues = upgrade.smart_upgrade([control], format=self.to_format, clean_up=True)
        self.assertLength(1, tried)
        self.assertEqual(tried[0], control)
        self.assertLength(1, worked)
        self.assertEqual(worked[0], control)
        self.assertLength(0, issues)
        self.assertPathExists('branch1')
        self.assertPathExists('branch1/.bzr')
        self.assertPathDoesNotExist('branch1/backup.bzr.~1~')
        self.assertEqual(control.open_repository()._format, self.to_format._repository_format)

    def make_repo_with_branches(self):
        repo = self.make_repository('repo', shared=True, format=self.from_format)
        controldir.ControlDir.create_branch_convenience('repo/branch1', format=self.from_format)
        b2 = controldir.ControlDir.create_branch_convenience('repo/branch2', format=self.from_format)
        return repo.controldir

    def test_upgrade_repo_with_branches(self):
        control = self.make_repo_with_branches()
        tried, worked, issues = upgrade.smart_upgrade([control], format=self.to_format)
        self.assertLength(3, tried)
        self.assertEqual(tried[0], control)
        self.assertLength(3, worked)
        self.assertEqual(worked[0], control)
        self.assertLength(0, issues)
        self.assertPathExists('repo/backup.bzr.~1~')
        self.assertPathExists('repo/branch1/backup.bzr.~1~')
        self.assertPathExists('repo/branch2/backup.bzr.~1~')
        self.assertEqual(control.open_repository()._format, self.to_format._repository_format)
        b1 = branch.Branch.open('repo/branch1')
        self.assertEqual(b1._format, self.to_format._branch_format)

    def test_upgrade_repo_with_branches_cleanup(self):
        control = self.make_repo_with_branches()
        tried, worked, issues = upgrade.smart_upgrade([control], format=self.to_format, clean_up=True)
        self.assertLength(3, tried)
        self.assertEqual(tried[0], control)
        self.assertLength(3, worked)
        self.assertEqual(worked[0], control)
        self.assertLength(0, issues)
        self.assertPathExists('repo')
        self.assertPathExists('repo/.bzr')
        self.assertPathDoesNotExist('repo/backup.bzr.~1~')
        self.assertPathDoesNotExist('repo/branch1/backup.bzr.~1~')
        self.assertPathDoesNotExist('repo/branch2/backup.bzr.~1~')
        self.assertEqual(control.open_repository()._format, self.to_format._repository_format)
        b1 = branch.Branch.open('repo/branch1')
        self.assertEqual(b1._format, self.to_format._branch_format)