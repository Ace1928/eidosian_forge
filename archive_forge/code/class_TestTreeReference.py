import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestTreeReference(TestCaseWithWorkingTree):

    def test_tree_reference_matches_inv(self):
        base = self.make_branch_and_tree('base')
        if base.branch.repository._format.supports_full_versioned_files:
            raise tests.TestNotApplicable('format does not support inventory deltas')
        if not base.supports_tree_reference():
            raise tests.TestNotApplicable("wt doesn't support nested trees")
        if base.has_versioned_directories():
            base.add(['subdir'], ['directory'])
            subdir = self.make_branch_and_tree('base/subdir')
        else:
            subdir = self.make_branch_and_tree('base/subdir')
            subdir.commit('')
            base.add(['subdir'], ['tree-reference'])
        self.addCleanup(base.lock_read().unlock)
        path, ie = next(base.iter_entries_by_dir(specific_files=['subdir']))
        self.assertEqual('subdir', path)
        self.assertEqual('tree-reference', ie.kind)