import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
class DiffBase(tests.TestCaseWithTransport):
    """Base class with common setup method"""

    def make_example_branch(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('hello', b'foo\n'), ('goodbye', b'baz\n')])
        tree.add(['hello'])
        tree.commit('setup')
        tree.add(['goodbye'])
        tree.commit('setup')
        return tree