from breezy import osutils, tests
class TestViewFileOperations(tests.TestCaseWithTransport):

    def make_abc_tree_with_ab_view(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b', 'c'])
        wt.views.set_view('my', ['a', 'b'])
        return wt

    def test_view_on_status(self):
        wt = self.make_abc_tree_with_ab_view()
        out, err = self.run_bzr('status')
        self.assertEqual('Ignoring files outside view. View is a, b\n', err)
        self.assertEqual('unknown:\n  a\n  b\n', out)

    def test_view_on_status_selected(self):
        wt = self.make_abc_tree_with_ab_view()
        out, err = self.run_bzr('status a')
        self.assertEqual('', err)
        self.assertEqual('unknown:\n  a\n', out)
        out, err = self.run_bzr('status c', retcode=3)
        self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_add(self):
        wt = self.make_abc_tree_with_ab_view()
        out, err = self.run_bzr('add')
        self.assertEqual('Ignoring files outside view. View is a, b\n', err)
        self.assertEqual('adding a\nadding b\n', out)

    def test_view_on_add_selected(self):
        wt = self.make_abc_tree_with_ab_view()
        out, err = self.run_bzr('add a')
        self.assertEqual('', err)
        self.assertEqual('adding a\n', out)
        out, err = self.run_bzr('add c', retcode=3)
        self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_diff(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('diff', retcode=1)
        self.assertEqual('*** Ignoring files outside view. View is a, b\n', err)

    def test_view_on_diff_selected(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('diff a', retcode=1)
        self.assertEqual('', err)
        self.assertStartsWith(out, "=== added file 'a'\n")
        out, err = self.run_bzr('diff c', retcode=3)
        self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_commit(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('commit -m "testing commit"')
        err_lines = err.splitlines()
        self.assertEqual('Ignoring files outside view. View is a, b', err_lines[0])
        self.assertStartsWith(err_lines[1], 'Committing to:')
        self.assertIn('added a', [err_lines[2], err_lines[3]])
        self.assertIn('added b', [err_lines[2], err_lines[3]])
        self.assertEqual('Committed revision 1.', err_lines[4])
        self.assertEqual('', out)

    def test_view_on_commit_selected(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('commit -m "file in view" a')
        err_lines = err.splitlines()
        self.assertStartsWith(err_lines[0], 'Committing to:')
        self.assertEqual('added a', err_lines[1])
        self.assertEqual('Committed revision 1.', err_lines[2])
        self.assertEqual('', out)
        out, err = self.run_bzr('commit -m "file out of view" c', retcode=3)
        self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_remove_selected(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('remove --keep a')
        self.assertEqual('removed a\n', err)
        self.assertEqual('', out)
        out, err = self.run_bzr('remove --keep c', retcode=3)
        self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_revert(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('revert')
        err_lines = err.splitlines()
        self.assertEqual('Ignoring files outside view. View is a, b', err_lines[0])
        self.assertEqual('-   a', err_lines[1])
        self.assertEqual('-   b', err_lines[2])
        self.assertEqual('', out)

    def test_view_on_revert_selected(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('revert a')
        self.assertEqual('-   a\n', err)
        self.assertEqual('', out)
        out, err = self.run_bzr('revert c', retcode=3)
        self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
        self.assertEqual('', out)

    def test_view_on_ls(self):
        wt = self.make_abc_tree_with_ab_view()
        self.run_bzr('add')
        out, err = self.run_bzr('ls')
        out_lines = out.splitlines()
        self.assertEqual('Ignoring files outside view. View is a, b\n', err)
        self.assertEqual('a', out_lines[0])
        self.assertEqual('b', out_lines[1])