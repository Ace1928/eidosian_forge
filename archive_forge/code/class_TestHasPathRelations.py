from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
class TestHasPathRelations(TestCaseWithTransport):

    def test__str__(self):
        t = self.make_branch_and_tree('.')
        matcher = HasPathRelations(t, [('a', 'b')])
        self.assertEqual('HasPathRelations(%r, %r)' % (t, [('a', 'b')]), str(matcher))

    def test_match(self):
        t = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/c'])
        t.add(['a', 'b', 'b/c'])
        self.assertThat(t, HasPathRelations(t, [('', ''), ('a', 'a'), ('b/', 'b/'), ('b/c', 'b/c')]))

    def test_mismatch(self):
        t = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b/', 'b/c'])
        t.add(['a', 'b', 'b/c'])
        mismatch = HasPathRelations(t, [('a', 'a')]).match(t)
        self.assertIsNot(None, mismatch)