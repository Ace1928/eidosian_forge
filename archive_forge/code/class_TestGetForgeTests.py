import os
from typing import List
from .. import forge as _mod_forge
from .. import registry, tests, urlutils
from ..forge import (Forge, MergeProposal, UnsupportedForge, determine_title,
class TestGetForgeTests(SampleForgeTestCase):

    def test_get_forge(self):
        tree = self.make_branch_and_tree('hosted/branch')
        self.assertIs(self.forge, get_forge(tree.branch, [self.forge]))
        self.assertIsInstance(get_forge(tree.branch), SampleForge)
        tree = self.make_branch_and_tree('blah')
        self.assertRaises(UnsupportedForge, get_forge, tree.branch)