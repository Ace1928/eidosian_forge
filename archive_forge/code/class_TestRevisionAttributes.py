from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
class TestRevisionAttributes(TestCaseWithRepository):
    """Test that revision attributes are correct."""

    def test_revision_accessors(self):
        """Make sure the values that come out of a revision are the
        same as the ones that go in.
        """
        tree1 = self.make_branch_and_tree('br1')
        if tree1.branch.repository._format.supports_custom_revision_properties:
            revprops = {'empty': '', 'value': 'one', 'unicode': 'µ', 'multiline': 'foo\nbar\n\n'}
        else:
            revprops = {}
        rev1 = tree1.commit(message='quux', allow_pointless=True, committer='jaq', revprops=revprops)
        self.assertEqual(tree1.branch.last_revision(), rev1)
        rev_a = tree1.branch.repository.get_revision(tree1.branch.last_revision())
        tree2 = self.make_branch_and_tree('br2')
        tree2.commit(message=rev_a.message, timestamp=rev_a.timestamp, timezone=rev_a.timezone, committer=rev_a.committer, rev_id=rev_a.revision_id if tree2.branch.repository._format.supports_setting_revision_ids else None, revprops=rev_a.properties, allow_pointless=True, strict=True, verbose=True)
        rev_b = tree2.branch.repository.get_revision(tree2.branch.last_revision())
        self.assertEqual(rev_a.message, rev_b.message)
        self.assertEqual(rev_a.timestamp, rev_b.timestamp)
        self.assertEqual(rev_a.timezone, rev_b.timezone)
        self.assertEqual(rev_a.committer, rev_b.committer)
        self.assertEqual(rev_a.revision_id, rev_b.revision_id)
        self.assertEqual(rev_a.properties, rev_b.properties)

    def test_zero_timezone(self):
        tree1 = self.make_branch_and_tree('br1')
        r1 = tree1.commit(message='quux', timezone=0)
        rev_a = tree1.branch.repository.get_revision(r1)
        self.assertEqual(0, rev_a.timezone)