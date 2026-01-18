import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
class TestUncommittedChanges(per_branch.TestCaseWithBranch):

    def setUp(self):
        super().setUp()
        if not self.branch_format.supports_store_uncommitted():
            raise tests.TestNotApplicable('Branch format does not support store_uncommitted')

    def bind(self, branch, master):
        try:
            branch.bind(master)
        except _mod_branch.BindingUnsupported:
            raise tests.TestNotApplicable('Branch cannot be bound.')

    def test_store_uncommitted(self):
        tree = self.make_branch_and_tree('b')
        branch = tree.branch
        creator = FakeShelfCreator(branch)
        with skip_if_storing_uncommitted_unsupported():
            self.assertIs(None, branch.get_unshelver(tree))
        branch.store_uncommitted(creator)
        self.assertIsNot(None, branch.get_unshelver(tree))

    def test_store_uncommitted_bound(self):
        tree = self.make_branch_and_tree('b')
        branch = tree.branch
        master = self.make_branch('master')
        self.bind(branch, master)
        creator = FakeShelfCreator(tree.branch)
        self.assertIs(None, tree.branch.get_unshelver(tree))
        self.assertIs(None, master.get_unshelver(tree))
        tree.branch.store_uncommitted(creator)
        self.assertIsNot(None, master.get_unshelver(tree))

    def test_store_uncommitted_already_stored(self):
        branch = self.make_branch('b')
        with skip_if_storing_uncommitted_unsupported():
            branch.store_uncommitted(FakeShelfCreator(branch))
        self.assertRaises(errors.ChangesAlreadyStored, branch.store_uncommitted, FakeShelfCreator(branch))

    def test_store_uncommitted_none(self):
        branch = self.make_branch('b')
        with skip_if_storing_uncommitted_unsupported():
            branch.store_uncommitted(FakeShelfCreator(branch))
        branch.store_uncommitted(None)
        self.assertIs(None, branch.get_unshelver(None))

    def test_get_unshelver(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('')
        self.build_tree_contents([('tree/file', b'contents1')])
        tree.add('file')
        with skip_if_storing_uncommitted_unsupported():
            tree.store_uncommitted()
        unshelver = tree.branch.get_unshelver(tree)
        self.assertIsNot(None, unshelver)

    def test_get_unshelver_bound(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('')
        self.build_tree_contents([('tree/file', b'contents1')])
        tree.add('file')
        with skip_if_storing_uncommitted_unsupported():
            tree.store_uncommitted()
        branch = self.make_branch('branch')
        self.bind(branch, tree.branch)
        unshelver = branch.get_unshelver(tree)
        self.assertIsNot(None, unshelver)