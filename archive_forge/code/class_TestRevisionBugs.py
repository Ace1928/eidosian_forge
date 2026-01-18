import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
class TestRevisionBugs(TestCase):
    """Tests for getting the bugs that a revision is linked to."""

    def test_no_bugs(self):
        r = revision.Revision('1')
        self.assertEqual([], list(r.iter_bugs()))

    def test_some_bugs(self):
        r = revision.Revision('1', properties={'bugs': bugtracker.encode_fixes_bug_urls([('http://example.com/bugs/1', 'fixed'), ('http://launchpad.net/bugs/1234', 'fixed')])})
        self.assertEqual([('http://example.com/bugs/1', bugtracker.FIXED), ('http://launchpad.net/bugs/1234', bugtracker.FIXED)], list(r.iter_bugs()))

    def test_no_status(self):
        r = revision.Revision('1', properties={'bugs': 'http://example.com/bugs/1'})
        self.assertRaises(bugtracker.InvalidLineInBugsProperty, list, r.iter_bugs())

    def test_too_much_information(self):
        r = revision.Revision('1', properties={'bugs': 'http://example.com/bugs/1 fixed bar'})
        self.assertRaises(bugtracker.InvalidLineInBugsProperty, list, r.iter_bugs())

    def test_invalid_status(self):
        r = revision.Revision('1', properties={'bugs': 'http://example.com/bugs/1 faxed'})
        self.assertRaises(bugtracker.InvalidBugStatus, list, r.iter_bugs())