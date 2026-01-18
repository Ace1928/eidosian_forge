from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
class TestBranchBuilder(tests.TestCaseWithMemoryTransport):

    def test_create(self):
        """Test the constructor api."""
        builder = BranchBuilder(self.get_transport().clone('foo'))

    def test_get_branch(self):
        """get_branch returns the created branch."""
        builder = BranchBuilder(self.get_transport().clone('foo'))
        branch = builder.get_branch()
        self.assertIsInstance(branch, _mod_branch.Branch)
        self.assertEqual(self.get_transport().clone('foo').base, branch.base)
        self.assertEqual((0, _mod_revision.NULL_REVISION), branch.last_revision_info())

    def test_format(self):
        """Making a BranchBuilder with a format option sets the branch type."""
        builder = BranchBuilder(self.get_transport(), format='dirstate-tags')
        branch = builder.get_branch()
        self.assertIsInstance(branch, _mod_bzrbranch.BzrBranch6)

    def test_build_one_commit(self):
        """doing build_commit causes a commit to happen."""
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id = builder.build_commit()
        branch = builder.get_branch()
        self.assertEqual((1, rev_id), branch.last_revision_info())
        self.assertEqual('commit 1', branch.repository.get_revision(branch.last_revision()).message)

    def test_build_commit_timestamp(self):
        """You can set a date when committing."""
        builder = self.make_branch_builder('foo')
        rev_id = builder.build_commit(timestamp=1236043340)
        branch = builder.get_branch()
        self.assertEqual((1, rev_id), branch.last_revision_info())
        rev = branch.repository.get_revision(branch.last_revision())
        self.assertEqual('commit 1', rev.message)
        self.assertEqual(1236043340, int(rev.timestamp))

    def test_build_two_commits(self):
        """The second commit has the right parents and message."""
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id1 = builder.build_commit()
        rev_id2 = builder.build_commit()
        branch = builder.get_branch()
        self.assertEqual((2, rev_id2), branch.last_revision_info())
        self.assertEqual('commit 2', branch.repository.get_revision(branch.last_revision()).message)
        self.assertEqual([rev_id1], branch.repository.get_revision(branch.last_revision()).parent_ids)

    def test_build_commit_parent_ids(self):
        """build_commit() takes a parent_ids argument."""
        builder = BranchBuilder(self.get_transport().clone('foo'))
        rev_id1 = builder.build_commit(parent_ids=[b'ghost'], allow_leftmost_as_ghost=True)
        rev_id2 = builder.build_commit(parent_ids=[])
        branch = builder.get_branch()
        self.assertEqual((1, rev_id2), branch.last_revision_info())
        self.assertEqual([b'ghost'], branch.repository.get_revision(rev_id1).parent_ids)