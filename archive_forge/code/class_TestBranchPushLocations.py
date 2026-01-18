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
class TestBranchPushLocations(per_branch.TestCaseWithBranch):

    def test_get_push_location_unset(self):
        self.assertEqual(None, self.get_branch().get_push_location())

    def test_get_push_location_exact(self):
        b = self.get_branch()
        config.LocationConfig.from_string('[{}]\npush_location=foo\n'.format(b.base), b.base, save=True)
        self.assertEqual('foo', self.get_branch().get_push_location())

    def test_set_push_location(self):
        branch = self.get_branch()
        branch.set_push_location('foo')
        self.assertEqual('foo', branch.get_push_location())