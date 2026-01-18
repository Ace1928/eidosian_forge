import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogMerges(TestLogWithLogCatcher):

    def setUp(self):
        super().setUp()
        self.make_branches_with_merges()

    def make_branches_with_merges(self):
        level0 = self.make_branch_and_tree('level0')
        self.wt_commit(level0, 'in branch level0')
        level1 = level0.controldir.sprout('level1').open_workingtree()
        self.wt_commit(level1, 'in branch level1')
        level2 = level1.controldir.sprout('level2').open_workingtree()
        self.wt_commit(level2, 'in branch level2')
        level1.merge_from_branch(level2.branch)
        self.wt_commit(level1, 'merge branch level2')
        level0.merge_from_branch(level1.branch)
        self.wt_commit(level0, 'merge branch level1')

    def test_merges_are_indented_by_level(self):
        self.run_bzr(['log', '-n0'], working_dir='level0')
        revnos_and_depth = [(r.revno, r.merge_depth) for r in self.get_captured_revisions()]
        self.assertEqual([('2', 0), ('1.1.2', 1), ('1.2.1', 2), ('1.1.1', 1), ('1', 0)], [(r.revno, r.merge_depth) for r in self.get_captured_revisions()])

    def test_force_merge_revisions_off(self):
        self.assertLogRevnos(['-n1'], ['2', '1'], working_dir='level0')

    def test_force_merge_revisions_on(self):
        self.assertLogRevnos(['-n0'], ['2', '1.1.2', '1.2.1', '1.1.1', '1'], working_dir='level0')

    def test_include_merged(self):
        expected = ['2', '1.1.2', '1.2.1', '1.1.1', '1']
        self.assertLogRevnos(['--include-merged'], expected, working_dir='level0')
        self.assertLogRevnos(['--include-merged'], expected, working_dir='level0')

    def test_force_merge_revisions_N(self):
        self.assertLogRevnos(['-n2'], ['2', '1.1.2', '1.1.1', '1'], working_dir='level0')

    def test_merges_single_merge_rev(self):
        self.assertLogRevnosAndDepths(['-n0', '-r1.1.2'], [('1.1.2', 0), ('1.2.1', 1)], working_dir='level0')

    def test_merges_partial_range(self):
        self.assertLogRevnosAndDepths(['-n0', '-r1.1.1..1.1.2'], [('1.1.2', 0), ('1.2.1', 1), ('1.1.1', 0)], working_dir='level0')

    def test_merges_partial_range_ignore_before_lower_bound(self):
        """Dont show revisions before the lower bound's merged revs"""
        self.assertLogRevnosAndDepths(['-n0', '-r1.1.2..2'], [('2', 0), ('1.1.2', 1), ('1.2.1', 2)], working_dir='level0')

    def test_omit_merges_with_sidelines(self):
        self.assertLogRevnos(['--omit-merges', '-n0'], ['1.2.1', '1.1.1', '1'], working_dir='level0')

    def test_omit_merges_without_sidelines(self):
        self.assertLogRevnos(['--omit-merges', '-n1'], ['1'], working_dir='level0')