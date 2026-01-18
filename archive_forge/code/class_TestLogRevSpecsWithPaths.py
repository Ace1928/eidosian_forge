import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogRevSpecsWithPaths(TestLogWithLogCatcher):

    def test_log_revno_n_path_wrong_namespace(self):
        self.make_linear_branch('branch1')
        self.make_linear_branch('branch2')
        self.run_bzr('log -r revno:2:branch1..revno:3:branch2', retcode=3)

    def test_log_revno_n_path_correct_order(self):
        self.make_linear_branch('branch2')
        self.assertLogRevnos(['-rrevno:1:branch2..revno:3:branch2'], ['3', '2', '1'])

    def test_log_revno_n_path(self):
        self.make_linear_branch('branch2')
        self.assertLogRevnos(['-rrevno:1:branch2'], ['1'])
        rev_props = self.log_catcher.revisions[0].rev.properties
        self.assertEqual('branch2', rev_props['branch-nick'])