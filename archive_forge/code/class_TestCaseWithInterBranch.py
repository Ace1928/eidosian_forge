from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
class TestCaseWithInterBranch(TestCaseWithTransport):

    def make_from_branch(self, relpath):
        return self.make_branch(relpath, format=self.branch_format_from._matchingcontroldir)

    def make_from_branch_and_memory_tree(self, relpath):
        """Create a branch on the default transport and a MemoryTree for it."""
        self.assertEqual(self.branch_format_from._matchingcontroldir.get_branch_format(), self.branch_format_from)
        return self.make_branch_and_memory_tree(relpath, format=self.branch_format_from._matchingcontroldir)

    def make_from_branch_and_tree(self, relpath):
        """Create a branch on the default transport and a working tree for it."""
        self.assertEqual(self.branch_format_from._matchingcontroldir.get_branch_format(), self.branch_format_from)
        return self.make_branch_and_tree(relpath, format=self.branch_format_from._matchingcontroldir)

    def make_from_branch_builder(self, relpath):
        self.assertEqual(self.branch_format_from._matchingcontroldir.get_branch_format(), self.branch_format_from)
        return branchbuilder.BranchBuilder(self.get_transport(relpath), format=self.branch_format_from._matchingcontroldir)

    def make_to_branch(self, relpath):
        self.assertEqual(self.branch_format_to._matchingcontroldir.get_branch_format(), self.branch_format_to)
        return self.make_branch(relpath, format=self.branch_format_to._matchingcontroldir)

    def make_to_branch_and_memory_tree(self, relpath):
        """Create a branch on the default transport and a MemoryTree for it."""
        self.assertEqual(self.branch_format_to._matchingcontroldir.get_branch_format(), self.branch_format_to)
        return self.make_branch_and_memory_tree(relpath, format=self.branch_format_to._matchingcontroldir)

    def make_to_branch_and_tree(self, relpath):
        """Create a branch on the default transport and a working tree for it."""
        self.assertEqual(self.branch_format_to._matchingcontroldir.get_branch_format(), self.branch_format_to)
        return self.make_branch_and_tree(relpath, format=self.branch_format_to._matchingcontroldir)

    def _sprout(self, origdir, to_url, format):
        if format.supports_workingtrees:
            newbranch = self.make_branch(to_url, format=format)
        else:
            newbranch = self.make_branch(to_url + '.branch', format=format)
        origbranch = origdir.open_branch()
        newbranch.repository.fetch(origbranch.repository)
        origbranch.copy_content_into(newbranch)
        if format.supports_workingtrees:
            wt = newbranch.controldir.create_workingtree()
        else:
            wt = newbranch.create_checkout(to_url, lightweight=True)
        return wt

    def sprout_to(self, origdir, to_url):
        """Sprout a bzrdir, using to_format for the new branch."""
        wt = self._sprout(origdir, to_url, self.branch_format_to._matchingcontroldir)
        self.assertEqual(wt.branch._format, self.branch_format_to)
        return wt.controldir

    def sprout_from(self, origdir, to_url):
        """Sprout a bzrdir, using from_format for the new bzrdir."""
        wt = self._sprout(origdir, to_url, self.branch_format_from._matchingcontroldir)
        self.assertEqual(wt.branch._format, self.branch_format_from)
        return wt.controldir