import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
class TestCheckoutDiff(TestDiff):

    def make_example_branch(self):
        tree = super().make_example_branch()
        tree = tree.branch.create_checkout('checkout')
        os.chdir('checkout')
        return tree

    def example_branch2(self):
        tree = super().example_branch2()
        os.mkdir('checkouts')
        tree = tree.branch.create_checkout('checkouts/branch1')
        os.chdir('checkouts')
        return tree

    def example_branches(self):
        branch1_tree, branch2_tree = super().example_branches()
        os.mkdir('checkouts')
        branch1_tree = branch1_tree.branch.create_checkout('checkouts/branch1')
        branch2_tree = branch2_tree.branch.create_checkout('checkouts/branch2')
        os.chdir('checkouts')
        return (branch1_tree, branch2_tree)