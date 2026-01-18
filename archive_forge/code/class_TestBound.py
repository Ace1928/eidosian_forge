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
class TestBound(per_branch.TestCaseWithBranch):

    def test_bind_unbind(self):
        branch = self.make_branch('1')
        branch2 = self.make_branch('2')
        try:
            branch.bind(branch2)
        except _mod_branch.BindingUnsupported:
            raise tests.TestNotApplicable('Format does not support binding')
        self.assertTrue(branch.unbind())
        self.assertFalse(branch.unbind())
        self.assertIs(None, branch.get_bound_location())

    def test_old_bound_location(self):
        branch = self.make_branch('branch1')
        try:
            self.assertIs(None, branch.get_old_bound_location())
        except errors.UpgradeRequired:
            raise tests.TestNotApplicable('Format does not store old bound locations')
        branch2 = self.make_branch('branch2')
        branch.bind(branch2)
        self.assertIs(None, branch.get_old_bound_location())
        branch.unbind()
        self.assertContainsRe(branch.get_old_bound_location(), '\\/branch2\\/$')

    def test_bind_diverged(self):
        tree_a = self.make_branch_and_tree('tree_a')
        tree_a.commit('rev1a')
        tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
        tree_a.commit('rev2a')
        tree_b.commit('rev2b')
        try:
            tree_b.branch.bind(tree_a.branch)
        except _mod_branch.BindingUnsupported:
            raise tests.TestNotApplicable('Format does not support binding')

    def test_unbind_clears_cached_master_branch(self):
        """b.unbind clears any cached value of b.get_master_branch."""
        master = self.make_branch('master')
        branch = self.make_branch('branch')
        try:
            branch.bind(master)
        except _mod_branch.BindingUnsupported:
            raise tests.TestNotApplicable('Format does not support binding')
        self.addCleanup(branch.lock_write().unlock)
        self.assertNotEqual(None, branch.get_master_branch())
        branch.unbind()
        self.assertEqual(None, branch.get_master_branch())

    def test_bind_clears_cached_master_branch(self):
        """b.bind clears any cached value of b.get_master_branch."""
        master1 = self.make_branch('master1')
        master2 = self.make_branch('master2')
        branch = self.make_branch('branch')
        try:
            branch.bind(master1)
        except _mod_branch.BindingUnsupported:
            raise tests.TestNotApplicable('Format does not support binding')
        self.addCleanup(branch.lock_write().unlock)
        self.assertNotEqual(None, branch.get_master_branch())
        branch.bind(master2)
        self.assertEqual('.', urlutils.relative_url(self.get_url('master2'), branch.get_master_branch().base))

    def test_set_bound_location_clears_cached_master_branch(self):
        """b.set_bound_location clears any cached value of b.get_master_branch.
        """
        master1 = self.make_branch('master1')
        master2 = self.make_branch('master2')
        branch = self.make_branch('branch')
        try:
            branch.bind(master1)
        except _mod_branch.BindingUnsupported:
            raise tests.TestNotApplicable('Format does not support binding')
        self.addCleanup(branch.lock_write().unlock)
        self.assertNotEqual(None, branch.get_master_branch())
        branch.set_bound_location(self.get_url('master2'))
        self.assertEqual('.', urlutils.relative_url(self.get_url('master2'), branch.get_master_branch().base))