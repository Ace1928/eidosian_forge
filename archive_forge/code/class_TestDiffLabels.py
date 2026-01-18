import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
class TestDiffLabels(DiffBase):

    def test_diff_label_removed(self):
        tree = super().make_example_branch()
        tree.remove('hello', keep_files=False)
        diff = self.run_bzr('diff', retcode=1)
        self.assertTrue("=== removed file 'hello'" in diff[0])

    def test_diff_label_added(self):
        tree = super().make_example_branch()
        self.build_tree_contents([('barbar', b'barbar')])
        tree.add('barbar')
        diff = self.run_bzr('diff', retcode=1)
        self.assertTrue("=== added file 'barbar'" in diff[0])

    def test_diff_label_modified(self):
        super().make_example_branch()
        self.build_tree_contents([('hello', b'barbar')])
        diff = self.run_bzr('diff', retcode=1)
        self.assertTrue("=== modified file 'hello'" in diff[0])

    def test_diff_label_renamed(self):
        tree = super().make_example_branch()
        tree.rename_one('hello', 'gruezi')
        diff = self.run_bzr('diff', retcode=1)
        self.assertTrue("=== renamed file 'hello' => 'gruezi'" in diff[0])