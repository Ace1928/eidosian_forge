from breezy import tests, workingtree
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack4
from breezy.bzr.knitrepo import RepositoryFormatKnit4
class TestSplit(tests.TestCaseWithTransport):

    def test_split(self):
        self.build_tree(['a/', 'a/b/', 'a/b/c'])
        wt = self.make_branch_and_tree('a', format='rich-root-pack')
        wt.add(['b', 'b/c'])
        wt.commit('rev1')
        self.run_bzr('split a/b')
        self.run_bzr_error(('.* is not versioned',), 'split q', working_dir='a')

    def test_split_repo_failure(self):
        repo = self.make_repository('branch', shared=True, format='knit')
        a_branch = repo.controldir.create_branch()
        self.build_tree(['a/', 'a/b/', 'a/b/c/', 'a/b/c/d'])
        wt = a_branch.create_checkout('a', lightweight=True)
        wt.add(['b', 'b/c', 'b/c/d'], ids=[b'b-id', b'c-id', b'd-id'])
        wt.commit('added files')
        self.run_bzr_error(('must upgrade your branch at .*a',), 'split a/b')

    def test_split_tree_failure(self):
        tree = self.make_branch_and_tree('tree', format='pack-0.92')
        self.build_tree(['tree/subtree/'])
        tree.add('subtree')
        tree.commit('added subtree')
        self.run_bzr_error(('must upgrade your branch at .*tree', 'rich roots'), 'split tree/subtree')

    def split_formats(self, format, repo_format):
        tree = self.make_branch_and_tree('rich-root', format=format)
        self.build_tree(['rich-root/a/'])
        tree.add('a')
        self.run_bzr(['split', 'rich-root/a'])
        subtree = workingtree.WorkingTree.open('rich-root/a')
        self.assertIsInstance(subtree.branch.repository._format, repo_format)

    def test_split_rich_root(self):
        self.split_formats('rich-root', RepositoryFormatKnit4)

    def test_split_rich_root_pack(self):
        self.split_formats('rich-root-pack', RepositoryFormatKnitPack4)