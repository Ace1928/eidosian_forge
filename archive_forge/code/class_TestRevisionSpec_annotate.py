import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_annotate(TestRevisionSpec):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('annotate-tree')
        self.build_tree_contents([('annotate-tree/file1', b'1\n')])
        self.tree.add('file1')
        self.tree.commit('r1', rev_id=b'r1')
        self.build_tree_contents([('annotate-tree/file1', b'2\n1\n')])
        self.tree.commit('r2', rev_id=b'r2')
        self.build_tree_contents([('annotate-tree/file1', b'2\n1\n3\n')])

    def test_as_revision_id_r1(self):
        self.assertAsRevisionId(b'r1', 'annotate:annotate-tree/file1:2')

    def test_as_revision_id_r2(self):
        self.assertAsRevisionId(b'r2', 'annotate:annotate-tree/file1:1')

    def test_as_revision_id_uncommitted(self):
        spec = RevisionSpec.from_string('annotate:annotate-tree/file1:3')
        e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
        self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/file1:3\\' does not exist in branch: .*\nLine 3 has not been committed.")

    def test_non_existent_line(self):
        spec = RevisionSpec.from_string('annotate:annotate-tree/file1:4')
        e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
        self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/file1:4\\' does not exist in branch: .*\nNo such line: 4")

    def test_invalid_line(self):
        spec = RevisionSpec.from_string('annotate:annotate-tree/file1:q')
        e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
        self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/file1:q\\' does not exist in branch: .*\nNo such line: q")

    def test_no_such_file(self):
        spec = RevisionSpec.from_string('annotate:annotate-tree/file2:1')
        e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
        self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/file2:1\\' does not exist in branch: .*\nFile 'file2' is not versioned")

    def test_no_such_file_with_colon(self):
        spec = RevisionSpec.from_string('annotate:annotate-tree/fi:le2:1')
        e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
        self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/fi:le2:1\\' does not exist in branch: .*\nFile 'fi:le2' is not versioned")