import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
class TestSwitchUncommitted(TestCaseWithTransport):

    def prepare(self):
        tree = self.make_branch_and_tree('orig')
        tree.commit('')
        tree.branch.controldir.sprout('new')
        checkout = tree.branch.create_checkout('checkout', lightweight=True)
        self.build_tree(['checkout/a'])
        self.assertPathExists('checkout/a')
        checkout.add('a')
        return checkout

    def test_store_and_restore_uncommitted(self):
        checkout = self.prepare()
        self.run_bzr(['switch', '--store', '-d', 'checkout', 'new'])
        self.build_tree(['checkout/b'])
        checkout.add('b')
        self.assertPathDoesNotExist('checkout/a')
        self.assertPathExists('checkout/b')
        self.run_bzr(['switch', '--store', '-d', 'checkout', 'orig'])
        self.assertPathExists('checkout/a')
        self.assertPathDoesNotExist('checkout/b')

    def test_does_not_store(self):
        self.prepare()
        self.run_bzr(['switch', '-d', 'checkout', 'new'])
        self.assertPathExists('checkout/a')

    def test_does_not_restore_changes(self):
        self.prepare()
        self.run_bzr(['switch', '--store', '-d', 'checkout', 'new'])
        self.assertPathDoesNotExist('checkout/a')
        self.run_bzr(['switch', '-d', 'checkout', 'orig'])
        self.assertPathDoesNotExist('checkout/a')