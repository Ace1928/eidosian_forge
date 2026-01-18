import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
class TestCheckout(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        tree = controldir.ControlDir.create_standalone_workingtree('branch')
        self.rev1 = tree.commit('1', allow_pointless=True)
        self.build_tree(['branch/added_in_2'])
        tree.add('added_in_2')
        self.rev2 = tree.commit('2')

    def test_checkout_makes_bound_branch(self):
        self.run_bzr('checkout branch checkout')
        source = controldir.ControlDir.open('branch')
        result = controldir.ControlDir.open('checkout')
        self.assertEqual(source.open_branch().controldir.root_transport.base, result.open_branch().get_bound_location())

    def test_checkout_light_makes_checkout(self):
        self.run_bzr('checkout --lightweight branch checkout')
        source = controldir.ControlDir.open('branch')
        result = controldir.ControlDir.open('checkout')
        self.assertEqual(source.open_branch().controldir.root_transport.base, result.open_branch().controldir.root_transport.base)

    def test_checkout_dash_r(self):
        out, err = self.run_bzr(['checkout', '-r', '-2', 'branch', 'checkout'])
        result = controldir.ControlDir.open('checkout')
        self.assertEqual([self.rev1], result.open_workingtree().get_parent_ids())
        self.assertPathDoesNotExist('checkout/added_in_2')

    def test_checkout_light_dash_r(self):
        out, err = self.run_bzr(['checkout', '--lightweight', '-r', '-2', 'branch', 'checkout'])
        result = controldir.ControlDir.open('checkout')
        self.assertEqual([self.rev1], result.open_workingtree().get_parent_ids())
        self.assertPathDoesNotExist('checkout/added_in_2')

    def test_checkout_into_empty_dir(self):
        self.make_controldir('checkout')
        out, err = self.run_bzr(['checkout', 'branch', 'checkout'])
        result = controldir.ControlDir.open('checkout')
        tree = result.open_workingtree()
        branch = result.open_branch()

    def test_checkout_reconstitutes_working_trees(self):
        os.mkdir('treeless-branch')
        branch = controldir.ControlDir.create_branch_convenience('treeless-branch', force_new_tree=False, format=bzrdir.BzrDirMetaFormat1())
        self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
        out, err = self.run_bzr('checkout treeless-branch')
        branch.controldir.open_workingtree()
        out, err = self.run_bzr('diff treeless-branch')
        branch = controldir.ControlDir.create_branch_convenience('.', force_new_tree=False, format=bzrdir.BzrDirMetaFormat1())
        self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
        out, err = self.run_bzr('checkout')
        branch.controldir.open_workingtree()
        out, err = self.run_bzr('diff')

    def _test_checkout_existing_dir(self, lightweight):
        source = self.make_branch_and_tree('source')
        self.build_tree_contents([('source/file1', b'content1'), ('source/file2', b'content2')])
        source.add(['file1', 'file2'])
        source.commit('added files')
        self.build_tree_contents([('target/', b''), ('target/file1', b'content1'), ('target/file2', b'content3')])
        cmd = ['checkout', 'source', 'target']
        if lightweight:
            cmd.append('--lightweight')
        self.run_bzr('checkout source target')
        self.assertPathExists('target/file2.moved')
        self.assertPathDoesNotExist('target/file1.moved')

    def test_checkout_existing_dir_heavy(self):
        self._test_checkout_existing_dir(False)

    def test_checkout_existing_dir_lightweight(self):
        self._test_checkout_existing_dir(True)

    def test_checkout_in_branch_with_r(self):
        branch = _mod_branch.Branch.open('branch')
        branch.controldir.destroy_workingtree()
        self.run_bzr('checkout -r 1', working_dir='branch')
        tree = workingtree.WorkingTree.open('branch')
        self.assertEqual(self.rev1, tree.last_revision())
        branch.controldir.destroy_workingtree()
        self.run_bzr('checkout -r 0', working_dir='branch')
        self.assertEqual(b'null:', tree.last_revision())

    def test_checkout_files_from(self):
        branch = _mod_branch.Branch.open('branch')
        self.run_bzr(['checkout', 'branch', 'branch2', '--files-from', 'branch'])

    def test_checkout_hardlink(self):
        self.requireFeature(HardlinkFeature(self.test_dir))
        source = self.make_branch_and_tree('source')
        self.build_tree(['source/file1'])
        source.add('file1')
        source.commit('added file')
        out, err = self.run_bzr('checkout source target --hardlink')
        source_stat = os.stat('source/file1')
        target_stat = os.stat('target/file1')
        self.assertEqual(source_stat, target_stat)

    def test_checkout_hardlink_files_from(self):
        self.requireFeature(HardlinkFeature(self.test_dir))
        source = self.make_branch_and_tree('source')
        self.build_tree(['source/file1'])
        source.add('file1')
        source.commit('added file')
        source.controldir.sprout('second')
        out, err = self.run_bzr('checkout source target --hardlink --files-from second')
        second_stat = os.stat('second/file1')
        target_stat = os.stat('target/file1')
        self.assertEqual(second_stat, target_stat)

    def test_colo_checkout(self):
        source = self.make_branch_and_tree('source', format='development-colo')
        self.build_tree(['source/file1'])
        source.add('file1')
        source.commit('added file')
        target = source.controldir.sprout('file:second,branch=somebranch', create_tree_if_local=False)
        out, err = self.run_bzr('checkout file:,branch=somebranch .', working_dir='second')
        self.assertEqual(target.open_branch(name='somebranch').user_url, target.get_branch_reference(name=''))