import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogFile(TestLogWithLogCatcher):

    def test_log_local_branch_file(self):
        """We should be able to log files in local treeless branches"""
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file'])
        tree.add('file')
        tree.commit('revision 1')
        tree.controldir.destroy_workingtree()
        self.run_bzr('log tree/file')

    def prepare_tree(self, complex=False):
        tree = self.make_branch_and_tree('parent')
        self.build_tree(['parent/file1', 'parent/file2', 'parent/file3'])
        tree.add('file1')
        tree.commit('add file1')
        tree.add('file2')
        tree.commit('add file2')
        tree.add('file3')
        tree.commit('add file3')
        child_tree = tree.controldir.sprout('child').open_workingtree()
        self.build_tree_contents([('child/file2', b'hello')])
        child_tree.commit(message='branch 1')
        tree.merge_from_branch(child_tree.branch)
        tree.commit(message='merge child branch')
        if complex:
            tree.remove('file2')
            tree.commit('remove file2')
            tree.rename_one('file3', 'file4')
            tree.commit('file3 is now called file4')
            tree.remove('file1')
            tree.commit('remove file1')
        os.chdir('parent')

    def test_log_file1(self):
        self.prepare_tree()
        self.assertLogRevnos(['-n0', 'file1'], ['1'])

    def test_log_file2(self):
        self.prepare_tree()
        self.assertLogRevnos(['-n0', 'file2'], ['4', '3.1.1', '2'])
        self.assertLogRevnos(['-n0', '-r3.1.1', 'file2'], ['3.1.1'])
        self.assertLogRevnos(['-n0', '-r4', 'file2'], ['4', '3.1.1'])
        self.assertLogRevnos(['-n0', '-r3..', 'file2'], ['4', '3.1.1'])
        self.assertLogRevnos(['-n0', '-r..3', 'file2'], ['2'])

    def test_log_file3(self):
        self.prepare_tree()
        self.assertLogRevnos(['-n0', 'file3'], ['3'])

    def test_log_file_historical_missing(self):
        self.prepare_tree(complex=True)
        err_msg = 'Path unknown at end or start of revision range: file2'
        err = self.run_bzr('log file2', retcode=3)[1]
        self.assertContainsRe(err, err_msg)

    def test_log_file_historical_end(self):
        self.prepare_tree(complex=True)
        self.assertLogRevnos(['-n0', '-r..4', 'file2'], ['4', '3.1.1', '2'])

    def test_log_file_historical_start(self):
        self.prepare_tree(complex=True)
        self.assertLogRevnos(['file1'], [])

    def test_log_file_renamed(self):
        """File matched against revision range, not current tree."""
        self.prepare_tree(complex=True)
        err_msg = 'Path unknown at end or start of revision range: file3'
        err = self.run_bzr('log file3', retcode=3)[1]
        self.assertContainsRe(err, err_msg)
        self.assertLogRevnos(['-r..4', 'file3'], ['3'])