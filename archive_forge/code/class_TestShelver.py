import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
class TestShelver(ShelfTestCase):

    def test_unexpected_prompt_failure(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        e = self.assertRaises(AssertionError, shelver.run)
        self.assertEqual('Unexpected prompt: Shelve?', str(e))

    def test_wrong_prompt_failure(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('foo', 0)
        e = self.assertRaises(AssertionError, shelver.run)
        self.assertEqual('Wrong prompt: Shelve?', str(e))

    def test_shelve_not_diff(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve?', 1)
        shelver.expect('Shelve?', 1)
        shelver.run()
        self.assertFileEqual(LINES_ZY, 'tree/foo')

    def test_shelve_diff_no(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve 2 change(s)?', 1)
        shelver.run()
        self.assertFileEqual(LINES_ZY, 'tree/foo')

    def test_shelve_diff(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve 2 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_one_diff(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve?', 1)
        shelver.expect('Shelve 1 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AY, 'tree/foo')

    def test_shelve_binary_change(self):
        tree = self.create_shelvable_tree()
        self.build_tree_contents([('tree/foo', b'\x00')])
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve binary changes?', 0)
        shelver.expect('Shelve 1 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_rename(self):
        tree = self.create_shelvable_tree()
        tree.rename_one('foo', 'bar')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve renaming "foo" => "bar"?', 0)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve?', 0)
        shelver.expect('Shelve 3 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_deletion(self):
        tree = self.create_shelvable_tree()
        os.unlink('tree/foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve removing file "foo"?', 0)
        shelver.expect('Shelve 1 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_creation(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('add tree root')
        self.build_tree(['tree/foo'])
        tree.add('foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve adding file "foo"?', 0)
        shelver.expect('Shelve 1 change(s)?', 0)
        shelver.run()
        self.assertPathDoesNotExist('tree/foo')

    def test_shelve_kind_change(self):
        tree = self.create_shelvable_tree()
        os.unlink('tree/foo')
        os.mkdir('tree/foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve changing "foo" from file to directory?', 0)
        shelver.expect('Shelve 1 change(s)?', 0)

    def test_shelve_modify_target(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tree = self.create_shelvable_tree()
        os.symlink('bar', 'tree/baz')
        tree.add('baz', ids=b'baz-id')
        tree.commit('Add symlink')
        os.unlink('tree/baz')
        os.symlink('vax', 'tree/baz')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve changing target of "baz" from "bar" to "vax"?', 0)
        shelver.expect('Shelve 1 change(s)?', 0)
        shelver.run()
        self.assertEqual('bar', os.readlink('tree/baz'))

    def test_shelve_finish(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve?', 2)
        shelver.expect('Shelve 2 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_quit(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree())
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve?', 3)
        self.assertRaises(errors.UserAbort, shelver.run)
        self.assertFileEqual(LINES_ZY, 'tree/foo')

    def test_shelve_all(self):
        self.create_shelvable_tree()
        shelver = ExpectShelver.from_args(sys.stdout, all=True, directory='tree')
        try:
            shelver.run()
        finally:
            shelver.finalize()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_filename(self):
        tree = self.create_shelvable_tree()
        self.build_tree(['tree/bar'])
        tree.add('bar')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), file_list=['bar'])
        self.addCleanup(shelver.finalize)
        shelver.expect('Shelve adding file "bar"?', 0)
        shelver.expect('Shelve 1 change(s)?', 0)
        shelver.run()

    def test_shelve_destroy(self):
        tree = self.create_shelvable_tree()
        shelver = shelf_ui.Shelver.from_args(sys.stdout, all=True, directory='tree', destroy=True)
        self.addCleanup(shelver.finalize)
        shelver.run()
        self.assertIs(None, tree.get_shelf_manager().last_shelf())
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    @staticmethod
    def shelve_all(tree, target_revision_id):
        tree.lock_write()
        try:
            target = tree.branch.repository.revision_tree(target_revision_id)
            shelver = shelf_ui.Shelver(tree, target, auto=True, auto_apply=True)
            try:
                shelver.run()
            finally:
                shelver.finalize()
        finally:
            tree.unlock()

    def test_shelve_old_root_preserved(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree1.commit('add root')
        tree1_root_id = tree1.path2id('')
        tree2 = self.make_branch_and_tree('tree2')
        rev2 = tree2.commit('add root')
        self.assertNotEqual(tree1_root_id, tree2.path2id(''))
        tree1.merge_from_branch(tree2.branch, from_revision=revision.NULL_REVISION)
        tree1.commit('merging in tree2')
        self.assertEqual(tree1_root_id, tree1.path2id(''))
        e = self.assertRaises(AssertionError, self.assertRaises, errors.InconsistentDelta, self.shelve_all, tree1, rev2)
        self.assertContainsRe('InconsistentDelta not raised', str(e))

    def test_shelve_split(self):
        outer_tree = self.make_branch_and_tree('outer')
        outer_tree.commit('Add root')
        inner_tree = self.make_branch_and_tree('outer/inner')
        rev2 = inner_tree.commit('Add root')
        outer_tree.subsume(inner_tree)
        self.expectFailure('Cannot shelve a join back to the inner tree.', self.assertRaises, AssertionError, self.assertRaises, ValueError, self.shelve_all, outer_tree, rev2)