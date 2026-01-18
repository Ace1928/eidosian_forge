import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
class TestApplyReporter(ShelfTestCase):

    def test_shelve_not_diff(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Apply change?', 1)
        shelver.expect('Apply change?', 1)
        shelver.run()
        self.assertFileEqual(LINES_ZY, 'tree/foo')

    def test_shelve_diff_no(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Apply change?', 0)
        shelver.expect('Apply change?', 0)
        shelver.expect('Apply 2 change(s)?', 1)
        shelver.run()
        self.assertFileEqual(LINES_ZY, 'tree/foo')

    def test_shelve_diff(self):
        tree = self.create_shelvable_tree()
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Apply change?', 0)
        shelver.expect('Apply change?', 0)
        shelver.expect('Apply 2 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_binary_change(self):
        tree = self.create_shelvable_tree()
        self.build_tree_contents([('tree/foo', b'\x00')])
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Apply binary changes?', 0)
        shelver.expect('Apply 1 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_rename(self):
        tree = self.create_shelvable_tree()
        tree.rename_one('foo', 'bar')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Rename "bar" => "foo"?', 0)
        shelver.expect('Apply change?', 0)
        shelver.expect('Apply change?', 0)
        shelver.expect('Apply 3 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_deletion(self):
        tree = self.create_shelvable_tree()
        os.unlink('tree/foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Add file "foo"?', 0)
        shelver.expect('Apply 1 change(s)?', 0)
        shelver.run()
        self.assertFileEqual(LINES_AJ, 'tree/foo')

    def test_shelve_creation(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('add tree root')
        self.build_tree(['tree/foo'])
        tree.add('foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Delete file "foo"?', 0)
        shelver.expect('Apply 1 change(s)?', 0)
        shelver.run()
        self.assertPathDoesNotExist('tree/foo')

    def test_shelve_kind_change(self):
        tree = self.create_shelvable_tree()
        os.unlink('tree/foo')
        os.mkdir('tree/foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Change "foo" from directory to a file?', 0)
        shelver.expect('Apply 1 change(s)?', 0)

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
        shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
        self.addCleanup(shelver.finalize)
        shelver.expect('Change target of "baz" from "vax" to "bar"?', 0)
        shelver.expect('Apply 1 change(s)?', 0)
        shelver.run()
        self.assertEqual('bar', os.readlink('tree/baz'))