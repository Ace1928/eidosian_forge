import os
import tarfile
import zipfile
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.per_tree import TestCaseWithTree
class GenericArchiveTests(TestCaseWithTree):

    def test_dir_invalid(self):
        work_a = self.make_branch_and_tree('wta')
        self.build_tree_contents([('wta/file', b'a\nb\nc\nd\n'), ('wta/dir', b'')])
        work_a.add('file')
        work_a.add('dir')
        work_a.commit('add file')
        tree_a = self.workingtree_to_test_tree(work_a)
        self.assertRaises(errors.NoSuchExportFormat, tree_a.archive, 'dir', 'foo')