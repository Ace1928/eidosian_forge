from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
class TestViewUI(TestCaseWithTransport):

    def test_view_command_help(self):
        out, err = self.run_bzr('help view')
        self.assertContainsRe(out, 'Manage filtered views')

    def test_define_view(self):
        wt = self.make_branch_and_tree('.')
        out, err = self.run_bzr('view a b c')
        self.assertEqual(out, "Using 'my' view: a, b, c\n")
        out, err = self.run_bzr('view e f --name foo')
        self.assertEqual(out, "Using 'foo' view: e, f\n")
        out, err = self.run_bzr('view p q')
        self.assertEqual(out, "Using 'foo' view: p, q\n")
        out, err = self.run_bzr('view r s --name my')
        self.assertEqual(out, "Using 'my' view: r, s\n")
        out, err = self.run_bzr('view a --name off', retcode=3)
        self.assertContainsRe(err, "Cannot change the 'off' pseudo view")

    def test_list_view(self):
        wt = self.make_branch_and_tree('.')
        out, err = self.run_bzr('view')
        self.assertEqual(out, 'No current view.\n')
        self.run_bzr('view a b c')
        out, err = self.run_bzr('view')
        self.assertEqual(out, "'my' view is: a, b, c\n")
        self.run_bzr('view e f --name foo')
        out, err = self.run_bzr('view --name my')
        self.assertEqual(out, "'my' view is: a, b, c\n")
        out, err = self.run_bzr('view --name foo')
        self.assertEqual(out, "'foo' view is: e, f\n")
        out, err = self.run_bzr('view --all')
        self.assertEqual(out.splitlines(), ['Views defined:', '=> foo                  e, f', '   my                   a, b, c'])
        out, err = self.run_bzr('view --name bar', retcode=3)
        self.assertContainsRe(err, 'No such view')

    def test_delete_view(self):
        wt = self.make_branch_and_tree('.')
        out, err = self.run_bzr('view --delete', retcode=3)
        self.assertContainsRe(err, 'No current view to delete')
        self.run_bzr('view a b c')
        out, err = self.run_bzr('view --delete')
        self.assertEqual(out, "Deleted 'my' view.\n")
        self.run_bzr('view e f --name foo')
        out, err = self.run_bzr('view --name foo --delete')
        self.assertEqual(out, "Deleted 'foo' view.\n")
        out, err = self.run_bzr('view --delete --all')
        self.assertEqual(out, 'Deleted all views.\n')
        out, err = self.run_bzr('view --delete --name bar', retcode=3)
        self.assertContainsRe(err, 'No such view')
        out, err = self.run_bzr('view --delete --switch x', retcode=3)
        self.assertContainsRe(err, 'Both --delete and --switch specified')
        out, err = self.run_bzr('view --delete a b c', retcode=3)
        self.assertContainsRe(err, 'Both --delete and a file list specified')

    def test_switch_view(self):
        wt = self.make_branch_and_tree('.')
        self.run_bzr('view a b c')
        self.run_bzr('view e f --name foo')
        out, err = self.run_bzr('view --switch my')
        self.assertEqual(out, "Using 'my' view: a, b, c\n")
        out, err = self.run_bzr('view --switch off')
        self.assertEqual(out, "Disabled 'my' view.\n")
        out, err = self.run_bzr('view --switch off', retcode=3)
        self.assertContainsRe(err, 'No current view to disable')
        out, err = self.run_bzr('view --switch x --all', retcode=3)
        self.assertContainsRe(err, 'Both --switch and --all specified')