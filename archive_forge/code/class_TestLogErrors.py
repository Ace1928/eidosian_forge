import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogErrors(TestLog):

    def test_log_zero_revspec(self):
        self.make_minimal_branch()
        self.run_bzr_error(['brz: ERROR: Logging revision 0 is invalid.'], ['log', '-r0'])

    def test_log_zero_begin_revspec(self):
        self.make_linear_branch()
        self.run_bzr_error(['brz: ERROR: Logging revision 0 is invalid.'], ['log', '-r0..2'])

    def test_log_zero_end_revspec(self):
        self.make_linear_branch()
        self.run_bzr_error(['brz: ERROR: Logging revision 0 is invalid.'], ['log', '-r-2..0'])

    def test_log_nonexistent_revno(self):
        self.make_minimal_branch()
        self.run_bzr_error(["brz: ERROR: Requested revision: '1234' does not exist in branch:"], ['log', '-r1234'])

    def test_log_nonexistent_dotted_revno(self):
        self.make_minimal_branch()
        self.run_bzr_error(["brz: ERROR: Requested revision: '123.123' does not exist in branch:"], ['log', '-r123.123'])

    def test_log_change_nonexistent_revno(self):
        self.make_minimal_branch()
        self.run_bzr_error(["brz: ERROR: Requested revision: '1234' does not exist in branch:"], ['log', '-c1234'])

    def test_log_change_nonexistent_dotted_revno(self):
        self.make_minimal_branch()
        self.run_bzr_error(["brz: ERROR: Requested revision: '123.123' does not exist in branch:"], ['log', '-c123.123'])

    def test_log_change_single_revno_only(self):
        self.make_minimal_branch()
        self.run_bzr_error(['brz: ERROR: Option --change does not accept revision ranges'], ['log', '--change', '2..3'])

    def test_log_change_incompatible_with_revision(self):
        self.run_bzr_error(['brz: ERROR: --revision and --change are mutually exclusive'], ['log', '--change', '2', '--revision', '3'])

    def test_log_nonexistent_file(self):
        self.make_minimal_branch()
        out, err = self.run_bzr('log does-not-exist', retcode=3)
        self.assertContainsRe(err, 'Path unknown at end or start of revision range: does-not-exist')

    def test_log_reversed_revspecs(self):
        self.make_linear_branch()
        self.run_bzr_error(('brz: ERROR: Start revision must be older than the end revision.\n',), ['log', '-r3..1'])

    def test_log_reversed_dotted_revspecs(self):
        self.make_merged_branch()
        self.run_bzr_error(('brz: ERROR: Start revision not found in history of end revision.\n',), 'log -r 1.1.1..1')

    def test_log_bad_message_re(self):
        """Bad --message argument gives a sensible message

        See https://bugs.launchpad.net/bzr/+bug/251352
        """
        self.make_minimal_branch()
        out, err = self.run_bzr(['log', '-m', '*'], retcode=3)
        self.assertContainsRe(err, 'ERROR.*Invalid pattern.*nothing to repeat')
        self.assertNotContainsRe(err, 'Unprintable exception')
        self.assertEqual(out, '')

    def test_log_unsupported_timezone(self):
        self.make_linear_branch()
        self.run_bzr_error(['brz: ERROR: Unsupported timezone format "foo", options are "utc", "original", "local".'], ['log', '--timezone', 'foo'])

    def test_log_exclude_ancestry_no_range(self):
        self.make_linear_branch()
        self.run_bzr_error(['brz: ERROR: --exclude-common-ancestry requires -r with two revisions'], ['log', '--exclude-common-ancestry'])

    def test_log_exclude_ancestry_single_revision(self):
        self.make_merged_branch()
        self.run_bzr_error(['brz: ERROR: --exclude-common-ancestry requires two different revisions'], ['log', '--exclude-common-ancestry', '-r1.1.1..1.1.1'])