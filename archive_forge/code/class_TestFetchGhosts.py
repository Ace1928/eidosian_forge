from .. import TestCaseWithTransport
class TestFetchGhosts(TestCaseWithTransport):

    def test_fetch_ghosts(self):
        self.run_bzr('init')
        self.run_bzr('fetch-ghosts .')

    def test_fetch_ghosts_with_saved(self):
        wt = self.make_branch_and_tree('.')
        wt.branch.set_parent('.')
        self.run_bzr('fetch-ghosts')

    def test_fetch_ghosts_more(self):
        self.run_bzr('init')
        with open('myfile', 'wb') as f:
            f.write(b'hello')
        self.run_bzr('add')
        self.run_bzr('commit -m hello')
        self.run_bzr('branch . my_branch')
        self.run_bzr('fetch-ghosts my_branch')