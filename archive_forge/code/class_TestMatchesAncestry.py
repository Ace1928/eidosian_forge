from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
class TestMatchesAncestry(TestCaseWithTransport):

    def test__str__(self):
        matcher = MatchesAncestry('A repository', b'arevid')
        self.assertEqual("MatchesAncestry(repository='A repository', revision_id=%r)" % (b'arevid',), str(matcher))

    def test_match(self):
        b = self.make_branch_builder('.')
        b.start_series()
        revid1 = b.build_commit()
        revid2 = b.build_commit()
        b.finish_series()
        branch = b.get_branch()
        m = MatchesAncestry(branch.repository, revid2)
        self.assertThat([revid2, revid1], m)
        self.assertThat([revid1, revid2], m)
        m = MatchesAncestry(branch.repository, revid1)
        self.assertThat([revid1], m)
        m = MatchesAncestry(branch.repository, b'unknown')
        self.assertThat([b'unknown'], m)

    def test_mismatch(self):
        b = self.make_branch_builder('.')
        b.start_series()
        revid1 = b.build_commit()
        revid2 = b.build_commit()
        b.finish_series()
        branch = b.get_branch()
        m = MatchesAncestry(branch.repository, revid1)
        mismatch = m.match([])
        self.assertIsNot(None, mismatch)
        self.assertEqual('mismatched ancestry for revision {!r} was [{!r}], expected []'.format(revid1, revid1), mismatch.describe())