import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
class TestMergeRevisionRange(tests.TestCaseWithTransport):
    scenarios = (('whole-tree', dict(context='.')), ('file-only', dict(context='a')))

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('.')
        self.tree.commit('initial commit')
        for f in ('a', 'b'):
            self.build_tree([f])
            self.tree.add(f)
            self.tree.commit('added ' + f)

    def test_merge_reversed_revision_range(self):
        self.run_bzr('merge -r 2..1 ' + self.context)
        self.assertPathDoesNotExist('a')
        self.assertPathExists('b')