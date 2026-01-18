from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
class TestSimpleAnnotate(tests.TestCaseWithTransport):
    """Annotate tests with no complex setup."""

    def _setup_edited_file(self, relpath='.'):
        """Create a tree with a locally edited file."""
        tree = self.make_branch_and_tree(relpath)
        file_relpath = joinpath(relpath, 'file')
        self.build_tree_contents([(file_relpath, b'foo\ngam\n')])
        tree.add('file')
        tree.commit('add file', committer='test@host', rev_id=b'rev1')
        self.build_tree_contents([(file_relpath, b'foo\nbar\ngam\n')])
        return tree

    def test_annotate_cmd_revspec_branch(self):
        tree = self._setup_edited_file('trunk')
        tree.branch.create_checkout(self.get_url('work'), lightweight=True)
        out, err = self.run_bzr(['annotate', 'file', '-r', 'branch:../trunk'], working_dir='work')
        self.assertEqual('', err)
        self.assertEqual('1   test@ho | foo\n            | gam\n', out)

    def test_annotate_edited_file(self):
        tree = self._setup_edited_file()
        self.overrideEnv('BRZ_EMAIL', 'current@host2')
        out, err = self.run_bzr('annotate file')
        self.assertEqual('1   test@ho | foo\n2?  current | bar\n1   test@ho | gam\n', out)

    def test_annotate_edited_file_no_default(self):
        override_whoami(self)
        tree = self._setup_edited_file()
        out, err = self.run_bzr('annotate file')
        self.assertEqual('1   test@ho | foo\n2?  local u | bar\n1   test@ho | gam\n', out)

    def test_annotate_edited_file_show_ids(self):
        tree = self._setup_edited_file()
        self.overrideEnv('BRZ_EMAIL', 'current@host2')
        out, err = self.run_bzr('annotate file --show-ids')
        self.assertEqual('    rev1 | foo\ncurrent: | bar\n    rev1 | gam\n', out)

    def _create_merged_file(self):
        """Create a file with a pending merge and local edit."""
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('file', b'foo\ngam\n')])
        tree.add('file')
        tree.commit('add file', rev_id=b'rev1', committer='test@host')
        self.build_tree_contents([('file', b'foo\nbar\ngam\n')])
        tree.commit('right', rev_id=b'rev1.1.1', committer='test@host')
        tree.pull(tree.branch, True, b'rev1')
        self.build_tree_contents([('file', b'foo\nbaz\ngam\n')])
        tree.commit('left', rev_id=b'rev2', committer='test@host')
        tree.merge_from_branch(tree.branch, b'rev1.1.1')
        self.build_tree_contents([('file', b'local\nfoo\nbar\nbaz\ngam\n')])
        return tree

    def test_annotated_edited_merged_file_revnos(self):
        wt = self._create_merged_file()
        out, err = self.run_bzr(['annotate', 'file'])
        email = config.extract_email_address(wt.branch.get_config_stack().get('email'))
        self.assertEqual('3?    %-7s | local\n1     test@ho | foo\n1.1.1 test@ho | bar\n2     test@ho | baz\n1     test@ho | gam\n' % email[:7], out)

    def test_annotated_edited_merged_file_ids(self):
        self._create_merged_file()
        out, err = self.run_bzr(['annotate', 'file', '--show-ids'])
        self.assertEqual('current: | local\n    rev1 | foo\nrev1.1.1 | bar\n    rev2 | baz\n    rev1 | gam\n', out)

    def test_annotate_empty_file(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', b'')])
        tree.add('empty')
        tree.commit('add empty file')
        out, err = self.run_bzr(['annotate', 'empty'])
        self.assertEqual('', out)

    def test_annotate_removed_file(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', b'')])
        tree.add('empty')
        tree.commit('add empty file')
        tree.remove('empty')
        tree.commit('remove empty file')
        out, err = self.run_bzr(['annotate', '-r1', 'empty'])
        self.assertEqual('', out)

    def test_annotate_empty_file_show_ids(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', b'')])
        tree.add('empty')
        tree.commit('add empty file')
        out, err = self.run_bzr(['annotate', '--show-ids', 'empty'])
        self.assertEqual('', out)

    def test_annotate_nonexistant_file(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file'])
        tree.add(['file'])
        tree.commit('add a file')
        out, err = self.run_bzr(['annotate', 'doesnotexist'], retcode=3)
        self.assertEqual('', out)
        self.assertEqual('brz: ERROR: doesnotexist is not versioned.\n', err)

    def test_annotate_without_workingtree(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('empty', b'')])
        tree.add('empty')
        tree.commit('add empty file')
        bzrdir = tree.branch.controldir
        bzrdir.destroy_workingtree()
        self.assertFalse(bzrdir.has_workingtree())
        out, err = self.run_bzr(['annotate', 'empty'])
        self.assertEqual('', out)

    def test_annotate_directory(self):
        """Test --directory option"""
        wt = self.make_branch_and_tree('a')
        self.build_tree_contents([('a/hello.txt', b'my helicopter\n')])
        wt.add(['hello.txt'])
        wt.commit('commit', committer='test@user')
        out, err = self.run_bzr(['annotate', '-d', 'a', 'hello.txt'])
        self.assertEqualDiff('1   test@us | my helicopter\n', out)