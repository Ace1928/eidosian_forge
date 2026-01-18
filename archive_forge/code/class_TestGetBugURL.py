from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class TestGetBugURL(TestCaseWithMemoryTransport):
    """Tests for bugtracker.get_bug_url"""

    class TransientTracker:
        """An transient tracker used for testing."""

        @classmethod
        def get(klass, abbreviation, branch):
            klass.log.append(('get', abbreviation, branch))
            if abbreviation != 'transient':
                return None
            return klass()

        def get_bug_url(self, bug_id):
            self.log.append(('get_bug_url', bug_id))
            return 'http://bugs.example.com/%s' % bug_id

    def setUp(self):
        super().setUp()
        self.tracker_type = TestGetBugURL.TransientTracker
        self.tracker_type.log = []
        bugtracker.tracker_registry.register('transient', self.tracker_type)
        self.addCleanup(bugtracker.tracker_registry.remove, 'transient')

    def test_get_bug_url_for_transient_tracker(self):
        branch = self.make_branch('some_branch')
        self.assertEqual('http://bugs.example.com/1234', bugtracker.get_bug_url('transient', branch, '1234'))
        self.assertEqual([('get', 'transient', branch), ('get_bug_url', '1234')], self.tracker_type.log)

    def test_unrecognized_abbreviation_raises_error(self):
        """If the abbreviation is unrecognized, then raise an error."""
        branch = self.make_branch('some_branch')
        self.assertRaises(bugtracker.UnknownBugTrackerAbbreviation, bugtracker.get_bug_url, 'xxx', branch, '1234')
        self.assertEqual([('get', 'xxx', branch)], self.tracker_type.log)