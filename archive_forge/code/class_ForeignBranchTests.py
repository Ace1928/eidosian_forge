from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
class ForeignBranchTests(TestCaseWithTransport):
    """Basic tests for foreign branch implementations.

    These tests mainly make sure that the implementation covers the required
    bits of the API and returns reasonable values.
    """
    branch_factory = None

    def make_empty_branch(self):
        return self.branch_factory.make_empty_branch(self.get_transport())

    def make_branch(self):
        return self.branch_factory.make_branch(self.get_transport())

    def test_set_parent(self):
        """Test that setting the parent works."""
        branch = self.make_branch()
        branch.set_parent('foobar')

    def test_set_push_location(self):
        """Test that setting the push location works."""
        branch = self.make_branch()
        branch.set_push_location('http://bar/bloe')

    def test_repr_type(self):
        branch = self.make_branch()
        self.assertIsInstance(repr(branch), str)

    def test_get_parent(self):
        """Test that getting the parent location works, and returns None."""
        branch = self.make_branch()
        self.assertIs(None, branch.get_parent())

    def test_get_push_location(self):
        """Test that getting the push location works, and returns None."""
        branch = self.make_branch()
        self.assertIs(None, branch.get_push_location())

    def test_attributes(self):
        """Check that various required attributes are present."""
        branch = self.make_branch()
        self.assertIsNot(None, getattr(branch, 'repository', None))
        self.assertIsNot(None, getattr(branch, 'mapping', None))
        self.assertIsNot(None, getattr(branch, '_format', None))
        self.assertIsNot(None, getattr(branch, 'base', None))

    def test__get_nick(self):
        """Make sure _get_nick is implemented and returns a string."""
        branch = self.make_branch()
        self.assertIsInstance(branch._get_nick(local=False), str)
        self.assertIsInstance(branch._get_nick(local=True), str)

    def test_null_revid_revno(self):
        """null: should return revno 0."""
        branch = self.make_branch()
        self.assertEqual(0, branch.revision_id_to_revno(NULL_REVISION))

    def test_get_stacked_on_url(self):
        """Test that get_stacked_on_url() behaves as expected.

        Inter-Format stacking doesn't work yet, so all foreign implementations
        should raise UnstackableBranchFormat at the moment.
        """
        branch = self.make_branch()
        self.assertRaises(UnstackableBranchFormat, branch.get_stacked_on_url)

    def test_get_physical_lock_status(self):
        branch = self.make_branch()
        self.assertFalse(branch.get_physical_lock_status())

    def test_last_revision_empty_branch(self):
        branch = self.make_empty_branch()
        self.assertEqual(NULL_REVISION, branch.last_revision())
        self.assertEqual(0, branch.revno())
        self.assertEqual((0, NULL_REVISION), branch.last_revision_info())