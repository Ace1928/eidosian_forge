import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
class TestSwitchDoesntOpenMasterBranch(TestCaseWithTransport):

    def test_switch_create_doesnt_open_master_branch(self):
        master = self.make_branch_and_tree('master')
        master.commit('one')
        checkout = master.branch.create_checkout('checkout')
        opened = []

        def open_hook(branch):
            name = branch.base.rstrip('/').rsplit('/', 1)[1]
            opened.append(name)
        branch.Branch.hooks.install_named_hook('open', open_hook, 'open_hook_logger')
        self.run_bzr('switch --create-branch -d checkout feature')
        self.assertEqual(1, opened.count('master'))