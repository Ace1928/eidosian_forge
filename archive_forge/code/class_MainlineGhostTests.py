import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class MainlineGhostTests(TestLogWithLogCatcher):

    def setUp(self):
        super().setUp()
        tree = self.make_branch_and_tree('')
        tree.set_parent_ids([b'spooky'], allow_leftmost_as_ghost=True)
        tree.add('')
        tree.commit('msg1', rev_id=b'rev1')
        tree.commit('msg2', rev_id=b'rev2')

    def test_log_range(self):
        self.assertLogRevnos(['-r1..2'], ['2', '1'])

    def test_log_norange(self):
        self.assertLogRevnos([], ['2', '1'])

    def test_log_range_open_begin(self):
        stdout, stderr = self.run_bzr(['log', '-r..2'], retcode=3)
        self.assertEqual(['2', '1'], [r.revno for r in self.get_captured_revisions()])
        self.assertEqual('brz: ERROR: Further revision history missing.\n', stderr)

    def test_log_range_open_end(self):
        self.assertLogRevnos(['-r1..'], ['2', '1'])