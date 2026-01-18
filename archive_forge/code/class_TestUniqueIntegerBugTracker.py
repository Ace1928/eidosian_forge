from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class TestUniqueIntegerBugTracker(TestCaseWithMemoryTransport):

    def test_appends_id_to_base_url(self):
        """The URL of a bug is the base URL joined to the identifier."""
        tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/foo')
        self.assertEqual('http://bugs.example.com/foo1234', tracker.get_bug_url('1234'))

    def test_returns_tracker_if_abbreviation_matches(self):
        """The get() method should return an instance of the tracker if the
        given abbreviation matches the tracker's abbreviated name.
        """
        tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
        branch = self.make_branch('some_branch')
        self.assertIs(tracker, tracker.get('xxx', branch))

    def test_returns_none_if_abbreviation_doesnt_match(self):
        """The get() method should return None if the given abbreviated name
        doesn't match the tracker's abbreviation.
        """
        tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
        branch = self.make_branch('some_branch')
        self.assertIs(None, tracker.get('yyy', branch))

    def test_doesnt_consult_branch(self):
        """A UniqueIntegerBugTracker shouldn't consult the branch for tracker
        information.
        """
        tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
        self.assertIs(tracker, tracker.get('xxx', None))
        self.assertIs(None, tracker.get('yyy', None))

    def test_check_bug_id_only_accepts_integers(self):
        """A UniqueIntegerBugTracker accepts integers as bug IDs."""
        tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
        tracker.check_bug_id('1234')

    def test_check_bug_id_doesnt_accept_non_integers(self):
        """A UniqueIntegerBugTracker rejects non-integers as bug IDs."""
        tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
        self.assertRaises(bugtracker.MalformedBugIdentifier, tracker.check_bug_id, 'red')