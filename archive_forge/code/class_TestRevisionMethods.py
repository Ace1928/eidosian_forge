import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
class TestRevisionMethods(TestCase):

    def test_get_summary(self):
        r = revision.Revision('1')
        r.message = 'a'
        self.assertEqual('a', r.get_summary())
        r.message = 'a\nb'
        self.assertEqual('a', r.get_summary())
        r.message = '\na\nb'
        self.assertEqual('a', r.get_summary())
        r.message = None
        self.assertEqual('', r.get_summary())

    def test_get_apparent_authors(self):
        r = revision.Revision('1')
        r.committer = 'A'
        self.assertEqual(['A'], r.get_apparent_authors())
        r.properties['author'] = 'B'
        self.assertEqual(['B'], r.get_apparent_authors())
        r.properties['authors'] = 'C\nD'
        self.assertEqual(['C', 'D'], r.get_apparent_authors())

    def test_get_apparent_authors_no_committer(self):
        r = revision.Revision('1')
        self.assertEqual([], r.get_apparent_authors())