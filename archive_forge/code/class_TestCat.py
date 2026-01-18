from ... import tests
from ...transport import memory
class TestCat(tests.TestCaseWithTransport):

    def test_cat(self):
        tree = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/a', b'foo\n')])
        tree.add('a')
        self.run_bzr(['cat', 'a'], retcode=3, working_dir='branch')
        tree.commit(message='1')
        self.build_tree_contents([('branch/a', b'baz\n')])
        self.assertEqual('foo\n', self.run_bzr(['cat', 'a'], working_dir='branch')[0])
        self.assertEqual(b'foo\n', self.run_brz_subprocess(['cat', 'a'], working_dir='branch')[0])
        tree.commit(message='2')
        self.assertEqual('baz\n', self.run_bzr(['cat', 'a'], working_dir='branch')[0])
        self.assertEqual('foo\n', self.run_bzr(['cat', 'a', '-r', '1'], working_dir='branch')[0])
        self.assertEqual('baz\n', self.run_bzr(['cat', 'a', '-r', '-1'], working_dir='branch')[0])
        rev_id = tree.branch.last_revision()
        self.assertEqual('baz\n', self.run_bzr(['cat', 'a', '-r', 'revid:%s' % rev_id.decode('utf-8')], working_dir='branch')[0])
        self.assertEqual('foo\n', self.run_bzr(['cat', 'branch/a', '-r', 'revno:1:branch'])[0])
        self.run_bzr(['cat', 'a'], retcode=3)
        self.run_bzr(['cat', 'a', '-r', 'revno:1:branch-that-does-not-exist'], retcode=3)

    def test_cat_different_id(self):
        """'cat' works with old and new files"""
        self.disable_missing_extensions_warning()
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a-rev-tree', b'foo\n'), ('c-rev', b'baz\n'), ('d-rev', b'bar\n'), ('e-rev', b'qux\n')])
        with tree.lock_write():
            tree.add(['a-rev-tree', 'c-rev', 'd-rev', 'e-rev'])
            tree.commit('add test files', rev_id=b'first')
            tree.flush()
            tree.remove(['d-rev'])
            tree.rename_one('a-rev-tree', 'b-tree')
            tree.rename_one('c-rev', 'a-rev-tree')
            tree.rename_one('e-rev', 'old-rev')
            self.build_tree_contents([('e-rev', b'new\n')])
            tree.add(['e-rev'])
        self.run_bzr_error(["^brz: ERROR: u?'b-tree' is not present in revision .+$"], 'cat b-tree --name-from-revision')
        out, err = self.run_bzr('cat d-rev')
        self.assertEqual('', err)
        self.assertEqual('bar\n', out)
        out, err = self.run_bzr('cat a-rev-tree --name-from-revision')
        self.assertEqual('foo\n', out)
        self.assertEqual('', err)
        out, err = self.run_bzr('cat a-rev-tree')
        self.assertEqual('baz\n', out)
        self.assertEqual('', err)
        out, err = self.run_bzr('cat e-rev -rrevid:first')
        self.assertEqual('qux\n', out)
        self.assertEqual('', err)

    def test_remote_cat(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['README'])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        url = self.get_readonly_url() + '/README'
        out, err = self.run_bzr(['cat', url])
        self.assertEqual('contents of README\n', out)

    def test_cat_branch_revspec(self):
        wt = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        wt = self.make_branch_and_tree('b')
        out, err = self.run_bzr(['cat', '-r', 'branch:../a', 'README'], working_dir='b')
        self.assertEqual('contents of a/README\n', out)

    def test_cat_filters(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['README'])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        url = self.get_readonly_url() + '/README'
        out, err = self.run_bzr(['cat', url])
        self.assertEqual('contents of README\n', out)
        out, err = self.run_bzr(['cat', '--filters', url])
        self.assertEqual('contents of README\n', out)

    def test_cat_filters_applied(self):
        from ...tree import Tree
        from ..test_filters import _stack_2
        wt = self.make_branch_and_tree('.')
        self.build_tree_contents([('README', b'junk\nline 1 of README\nline 2 of README\n')])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        url = self.get_readonly_url() + '/README'
        real_content_filter_stack = Tree._content_filter_stack

        def _custom_content_filter_stack(tree, path=None, file_id=None):
            return _stack_2
        Tree._content_filter_stack = _custom_content_filter_stack
        try:
            out, err = self.run_bzr(['cat', url, '--filters'])
            self.assertEqual('LINE 1 OF readme\nLINE 2 OF readme\n', out)
            self.assertEqual('', err)
        finally:
            Tree._content_filter_stack = real_content_filter_stack

    def test_cat_no_working_tree(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['README'])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        wt.branch.controldir.destroy_workingtree()
        url = self.get_readonly_url() + '/README'
        out, err = self.run_bzr(['cat', url])
        self.assertEqual('contents of README\n', out)

    def test_cat_nonexistent_branch(self):
        self.vfs_transport_factory = memory.MemoryServer
        self.run_bzr_error(['^brz: ERROR: Not a branch'], ['cat', self.get_url()])

    def test_cat_directory(self):
        wt = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        out, err = self.run_bzr(['cat', '--directory=a', 'README'])
        self.assertEqual('contents of a/README\n', out)

    def test_cat_remote_directory(self):
        wt = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        wt.add('README')
        wt.commit('Making sure there is a basis_tree available')
        url = self.get_readonly_url() + '/a'
        out, err = self.run_bzr(['cat', '-d', url, 'README'])
        self.assertEqual('contents of a/README\n', out)