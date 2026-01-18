import os
from breezy import bedding, tests, workingtree
class TestLogFormats(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        conf_path = bedding.config_path()
        if os.path.isfile(conf_path):
            self.fail('%s exists' % conf_path)
        bedding.ensure_config_dir_exists()
        with open(conf_path, 'wb') as f:
            f.write(b'[DEFAULT]\nemail=Joe Foo <joe@foo.com>\nlog_format=line\n')

    def _make_simple_branch(self, relpath='.'):
        wt = self.make_branch_and_tree(relpath)
        wt.commit('first revision')
        wt.commit('second revision')
        return wt

    def test_log_default_format(self):
        self._make_simple_branch()
        log = self.run_bzr('log')[0]
        self.assertEqual(2, len(log.splitlines()))

    def test_log_format_arg(self):
        self._make_simple_branch()
        log = self.run_bzr(['log', '--log-format', 'short'])[0]

    def test_missing_default_format(self):
        wt = self._make_simple_branch('a')
        self.run_bzr(['branch', 'a', 'b'])
        wt.commit('third revision')
        wt.commit('fourth revision')
        missing = self.run_bzr('missing', retcode=1, working_dir='b')[0]
        self.assertEqual(4, len(missing.splitlines()))

    def test_missing_format_arg(self):
        wt = self._make_simple_branch('a')
        self.run_bzr(['branch', 'a', 'b'])
        wt.commit('third revision')
        wt.commit('fourth revision')
        missing = self.run_bzr(['missing', '--log-format', 'short'], retcode=1, working_dir='b')[0]
        self.assertEqual(8, len(missing.splitlines()))

    def test_logformat_gnu_changelog(self):
        wt = self.make_branch_and_tree('.')
        wt.commit('first revision', timestamp=1236045060, timezone=0)
        log, err = self.run_bzr(['log', '--log-format', 'gnu-changelog', '--timezone=utc'])
        self.assertEqual('', err)
        expected = '2009-03-03  Joe Foo  <joe@foo.com>\n\n\tfirst revision\n\n'
        self.assertEqualDiff(expected, log)

    def test_logformat_line_wide(self):
        """Author field should get larger for column widths over 80"""
        wt = self.make_branch_and_tree('.')
        wt.commit('revision with a long author', committer='Person with long name SENTINEL')
        log, err = self.run_bzr('log --line')
        self.assertNotContainsString(log, 'SENTINEL')
        self.overrideEnv('BRZ_COLUMNS', '116')
        log, err = self.run_bzr('log --line')
        self.assertContainsString(log, 'SENT...')
        self.overrideEnv('BRZ_COLUMNS', '0')
        log, err = self.run_bzr('log --line')
        self.assertContainsString(log, 'SENTINEL')