import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogDiff(TestLogWithLogCatcher):

    def setUp(self):
        super().setUp()
        self.make_branch_with_diffs()

    def make_branch_with_diffs(self):
        level0 = self.make_branch_and_tree('level0')
        self.build_tree(['level0/file1', 'level0/file2'])
        level0.add('file1')
        level0.add('file2')
        self.wt_commit(level0, 'in branch level0')
        level1 = level0.controldir.sprout('level1').open_workingtree()
        self.build_tree_contents([('level1/file2', b'hello\n')])
        self.wt_commit(level1, 'in branch level1')
        level0.merge_from_branch(level1.branch)
        self.wt_commit(level0, 'merge branch level1')

    def _diff_file1_revno1(self):
        return b"=== added file 'file1'\n--- file1\t1970-01-01 00:00:00 +0000\n+++ file1\t2005-11-22 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+contents of level0/file1\n\n"

    def _diff_file2_revno2(self):
        return b"=== modified file 'file2'\n--- file2\t2005-11-22 00:00:00 +0000\n+++ file2\t2005-11-22 00:00:01 +0000\n@@ -1,1 +1,1 @@\n-contents of level0/file2\n+hello\n\n"

    def _diff_file2_revno1_1_1(self):
        return b"=== modified file 'file2'\n--- file2\t2005-11-22 00:00:00 +0000\n+++ file2\t2005-11-22 00:00:01 +0000\n@@ -1,1 +1,1 @@\n-contents of level0/file2\n+hello\n\n"

    def _diff_file2_revno1(self):
        return b"=== added file 'file2'\n--- file2\t1970-01-01 00:00:00 +0000\n+++ file2\t2005-11-22 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+contents of level0/file2\n\n"

    def assertLogRevnosAndDiff(self, args, expected, working_dir='.'):
        self.run_bzr(['log', '-p'] + args, working_dir=working_dir)
        expected_revnos_and_depths = [(revno, depth) for revno, depth, diff in expected]
        self.assertEqual(expected_revnos_and_depths, [(r.revno, r.merge_depth) for r in self.get_captured_revisions()])
        fmt = 'In revno %s\n%s'
        for expected_rev, actual_rev in zip(expected, self.get_captured_revisions()):
            revno, depth, expected_diff = expected_rev
            actual_diff = actual_rev.diff
            self.assertEqualDiff(fmt % (revno, expected_diff), fmt % (revno, actual_diff))

    def test_log_diff_with_merges(self):
        self.assertLogRevnosAndDiff(['-n0'], [('2', 0, self._diff_file2_revno2()), ('1.1.1', 1, self._diff_file2_revno1_1_1()), ('1', 0, self._diff_file1_revno1() + self._diff_file2_revno1())], working_dir='level0')

    def test_log_diff_file1(self):
        self.assertLogRevnosAndDiff(['-n0', 'file1'], [('1', 0, self._diff_file1_revno1())], working_dir='level0')

    def test_log_diff_file2(self):
        self.assertLogRevnosAndDiff(['-n1', 'file2'], [('2', 0, self._diff_file2_revno2()), ('1', 0, self._diff_file2_revno1())], working_dir='level0')