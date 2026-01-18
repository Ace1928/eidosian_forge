from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
class FindPreviousPathsTests(TestCaseWithTransport):

    def test_new(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/b'])
        tree.add(['b'])
        revid1 = tree.commit('first')
        tree1 = tree.branch.repository.revision_tree(revid1)
        tree0 = tree.branch.repository.revision_tree(revision.NULL_REVISION)
        self.assertEqual({'b': None}, find_previous_paths(tree1, tree0, ['b']))

    def test_find_previous_paths(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/b'])
        tree.add(['b'])
        revid1 = tree.commit('first')
        tree1 = tree.branch.repository.revision_tree(revid1)
        tree.rename_one('b', 'c')
        self.build_tree(['tree/b'])
        tree.add(['b'])
        revid2 = tree.commit('second')
        tree2 = tree.branch.repository.revision_tree(revid2)
        self.assertEqual({'c': 'b', 'b': None}, find_previous_paths(tree2, tree1, ['b', 'c']))