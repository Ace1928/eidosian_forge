from breezy import osutils, tests
class TestViewTreeOperations(tests.TestCaseWithTransport):

    def make_abc_tree_and_clone_with_ab_view(self):
        wt1 = self.make_branch_and_tree('tree_1')
        self.build_tree(['tree_1/a', 'tree_1/b', 'tree_1/c'])
        wt1.add(['a', 'b', 'c'])
        wt1.commit('adding a b c')
        wt2 = wt1.controldir.sprout('tree_2').open_workingtree()
        wt2.views.set_view('my', ['a', 'b'])
        self.build_tree_contents([('tree_1/a', b'changed a\n'), ('tree_1/c', b'changed c\n')])
        wt1.commit('changing a c')
        return (wt1, wt2)

    def test_view_on_pull(self):
        tree_1, tree_2 = self.make_abc_tree_and_clone_with_ab_view()
        out, err = self.run_bzr('pull -d tree_2 tree_1')
        self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\n", err)
        self.assertEqualDiff('Now on revision 2.\n', out)

    def test_view_on_update(self):
        tree_1, tree_2 = self.make_abc_tree_and_clone_with_ab_view()
        self.run_bzr('bind ../tree_1', working_dir='tree_2')
        out, err = self.run_bzr('update', working_dir='tree_2')
        self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\n" % osutils.pathjoin(self.test_dir, 'tree_1'), err)
        self.assertEqual('', out)

    def test_view_on_merge(self):
        tree_1, tree_2 = self.make_abc_tree_and_clone_with_ab_view()
        out, err = self.run_bzr('merge -d tree_2 tree_1')
        self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\n", err)
        self.assertEqual('', out)