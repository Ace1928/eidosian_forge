from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class ErrorsTest(TestCaseWithMemoryTransport):

    def test_unknown_bug_tracker_abbreviation(self):
        """Test the formatting of UnknownBugTrackerAbbreviation."""
        branch = self.make_branch('some_branch')
        error = bugtracker.UnknownBugTrackerAbbreviation('xxx', branch)
        self.assertEqual('Cannot find registered bug tracker called xxx on %s' % branch, str(error))

    def test_malformed_bug_identifier(self):
        """Test the formatting of MalformedBugIdentifier."""
        error = bugtracker.MalformedBugIdentifier('bogus', 'reason for bogosity')
        self.assertEqual('Did not understand bug identifier bogus: reason for bogosity. See "brz help bugs" for more information on this feature.', str(error))

    def test_incorrect_url(self):
        err = bugtracker.InvalidBugTrackerURL('foo', 'http://bug.example.com/')
        self.assertEqual('The URL for bug tracker "foo" doesn\'t contain {id}: http://bug.example.com/', str(err))