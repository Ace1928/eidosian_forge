from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
class Test1To2Fetch(TestCaseWithTransport):
    """Tests for Model1To2 failure modes"""

    def make_tree_and_repo(self):
        self.tree = self.make_branch_and_tree('tree', format='pack-0.92')
        self.repo = self.make_repository('rich-repo', format='rich-root-pack')
        self.repo.lock_write()
        self.addCleanup(self.repo.unlock)

    def do_fetch_order_test(self, first, second):
        """Test that fetch works no matter what the set order of revision is.

        This test depends on the order of items in a set, which is
        implementation-dependant, so we test A, B and then B, A.
        """
        self.make_tree_and_repo()
        self.tree.commit('Commit 1', rev_id=first)
        self.tree.commit('Commit 2', rev_id=second)
        self.repo.fetch(self.tree.branch.repository, second)

    def test_fetch_order_AB(self):
        """See do_fetch_order_test"""
        self.do_fetch_order_test(b'A', b'B')

    def test_fetch_order_BA(self):
        """See do_fetch_order_test"""
        self.do_fetch_order_test(b'B', b'A')

    def get_parents(self, file_id, revision_id):
        self.repo.lock_read()
        try:
            parent_map = self.repo.texts.get_parent_map([(file_id, revision_id)])
            return parent_map[file_id, revision_id]
        finally:
            self.repo.unlock()

    def test_fetch_ghosts(self):
        self.make_tree_and_repo()
        self.tree.commit('first commit', rev_id=b'left-parent')
        self.tree.add_parent_tree_id(b'ghost-parent')
        fork = self.tree.controldir.sprout('fork', b'null:').open_workingtree()
        fork.commit('not a ghost', rev_id=b'not-ghost-parent')
        self.tree.branch.repository.fetch(fork.branch.repository, b'not-ghost-parent')
        self.tree.add_parent_tree_id(b'not-ghost-parent')
        self.tree.commit('second commit', rev_id=b'second-id')
        self.repo.fetch(self.tree.branch.repository, b'second-id')
        root_id = self.tree.path2id('')
        self.assertEqual(((root_id, b'left-parent'), (root_id, b'not-ghost-parent')), self.get_parents(root_id, b'second-id'))

    def make_two_commits(self, change_root, fetch_twice):
        self.make_tree_and_repo()
        self.tree.commit('first commit', rev_id=b'first-id')
        if change_root:
            self.tree.set_root_id(b'unique-id')
        self.tree.commit('second commit', rev_id=b'second-id')
        if fetch_twice:
            self.repo.fetch(self.tree.branch.repository, b'first-id')
        self.repo.fetch(self.tree.branch.repository, b'second-id')

    def test_fetch_changed_root(self):
        self.make_two_commits(change_root=True, fetch_twice=False)
        self.assertEqual((), self.get_parents(b'unique-id', b'second-id'))

    def test_two_fetch_changed_root(self):
        self.make_two_commits(change_root=True, fetch_twice=True)
        self.assertEqual((), self.get_parents(b'unique-id', b'second-id'))

    def test_two_fetches(self):
        self.make_two_commits(change_root=False, fetch_twice=True)
        self.assertEqual(((b'TREE_ROOT', b'first-id'),), self.get_parents(b'TREE_ROOT', b'second-id'))