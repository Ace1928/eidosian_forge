import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
class ExportTest:

    def prepare_export(self):
        work_a = self.make_branch_and_tree('wta')
        self.build_tree_contents([('wta/file', b'a\nb\nc\nd\n'), ('wta/dir', b'')])
        work_a.add('file')
        work_a.add('dir')
        work_a.commit('add file')
        tree_a = self.workingtree_to_test_tree(work_a)
        export(tree_a, 'output', self.exporter)

    def prepare_symlink_export(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        work_a = self.make_branch_and_tree('wta')
        os.symlink('target', 'wta/link')
        work_a.add('link')
        work_a.commit('add link')
        tree_a = self.workingtree_to_test_tree(work_a)
        export(tree_a, 'output', self.exporter)

    def test_export(self):
        self.prepare_export()
        names = self.get_export_names()
        self.assertIn('output/file', names)
        self.assertIn('output/dir', names)

    def test_export_symlink(self):
        self.prepare_symlink_export()
        names = self.get_export_names()
        self.assertIn('output/link', names)

    def prepare_nested_export(self, recurse_nested):
        tree = self.make_branch_and_tree('dir')
        self.build_tree(['dir/a'])
        tree.add('a')
        tree.commit('1')
        subtree = self.make_branch_and_tree('dir/subdir')
        self.build_tree(['dir/subdir/b'])
        subtree.add('b')
        subtree.commit('1a')
        try:
            tree.add_reference(subtree)
        except UnsupportedOperation:
            raise TestNotApplicable('format does not supported nested trees')
        tree.commit('2')
        export(tree, 'output', self.exporter, recurse_nested=recurse_nested)

    def test_export_nested_recurse(self):
        self.prepare_nested_export(True)
        names = self.get_export_names()
        self.assertIn('output/subdir/b', names)

    def test_export_nested_nonrecurse(self):
        self.prepare_nested_export(False)
        names = self.get_export_names()
        self.assertNotIn('output/subdir/b', names)