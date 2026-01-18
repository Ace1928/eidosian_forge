import os
import tarfile
import zipfile
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.per_tree import TestCaseWithTree
class TestZip(ArchiveTests, TestCaseWithTree):
    format = 'zip'

    def get_export_names(self, path):
        zf = zipfile.ZipFile(path)
        try:
            return zf.namelist()
        finally:
            zf.close()

    def test_export_symlink(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        work_a = self.make_branch_and_tree('wta')
        os.symlink('target', 'wta/link')
        work_a.add('link')
        work_a.commit('add link')
        tree_a = self.workingtree_to_test_tree(work_a)
        output_path = 'output'
        with open(output_path, 'wb') as f:
            f.writelines(tree_a.archive(self.format, output_path))
        names = self.get_export_names(output_path)
        self.assertIn('link.lnk', names)