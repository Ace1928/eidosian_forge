from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
class TestRevisionHistoryMatches(TestCaseWithTransport):

    def test_empty(self):
        tree = self.make_branch_and_tree('.')
        matcher = RevisionHistoryMatches([])
        self.assertIs(None, matcher.match(tree.branch))

    def test_matches(self):
        tree = self.make_branch_and_tree('.')
        tree.commit('msg1', rev_id=b'a')
        tree.commit('msg2', rev_id=b'b')
        matcher = RevisionHistoryMatches([b'a', b'b'])
        self.assertIs(None, matcher.match(tree.branch))

    def test_mismatch(self):
        tree = self.make_branch_and_tree('.')
        tree.commit('msg1', rev_id=b'a')
        tree.commit('msg2', rev_id=b'b')
        matcher = RevisionHistoryMatches([b'a', b'b', b'c'])
        self.assertEqual({"[b'a', b'b']", "[b'a', b'b', b'c']"}, set(matcher.match(tree.branch).describe().split(' != ')))