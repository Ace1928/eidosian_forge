import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
class TestUnshelver(tests.TestCaseWithTransport):

    def create_tree_with_shelf(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        try:
            self.build_tree_contents([('tree/foo', LINES_AJ)])
            tree.add('foo', ids=b'foo-id')
            tree.commit('added foo')
            self.build_tree_contents([('tree/foo', LINES_ZY)])
            shelver = shelf_ui.Shelver(tree, tree.basis_tree(), auto_apply=True, auto=True)
            try:
                shelver.run()
            finally:
                shelver.finalize()
        finally:
            tree.unlock()
        return tree

    def test_unshelve(self):
        tree = self.create_tree_with_shelf()
        tree.lock_write()
        self.addCleanup(tree.unlock)
        manager = tree.get_shelf_manager()
        shelf_ui.Unshelver(tree, manager, 1, True, True, True).run()
        self.assertFileEqual(LINES_ZY, 'tree/foo')

    def test_unshelve_args(self):
        tree = self.create_tree_with_shelf()
        unshelver = shelf_ui.Unshelver.from_args(directory='tree')
        try:
            unshelver.run()
        finally:
            unshelver.tree.unlock()
        self.assertFileEqual(LINES_ZY, 'tree/foo')
        self.assertIs(None, tree.get_shelf_manager().last_shelf())

    def test_unshelve_args_dry_run(self):
        tree = self.create_tree_with_shelf()
        unshelver = shelf_ui.Unshelver.from_args(directory='tree', action='dry-run')
        try:
            unshelver.run()
        finally:
            unshelver.tree.unlock()
        self.assertFileEqual(LINES_AJ, 'tree/foo')
        self.assertEqual(1, tree.get_shelf_manager().last_shelf())

    def test_unshelve_args_preview(self):
        tree = self.create_tree_with_shelf()
        write_diff_to = BytesIO()
        unshelver = shelf_ui.Unshelver.from_args(directory='tree', action='preview', write_diff_to=write_diff_to)
        try:
            unshelver.run()
        finally:
            unshelver.tree.unlock()
        self.assertFileEqual(LINES_AJ, 'tree/foo')
        self.assertEqual(1, tree.get_shelf_manager().last_shelf())
        diff = write_diff_to.getvalue()
        expected = dedent('            @@ -1,4 +1,4 @@\n            -a\n            +z\n             b\n             c\n             d\n            @@ -7,4 +7,4 @@\n             g\n             h\n             i\n            -j\n            +y\n\n            ')
        self.assertEqualDiff(expected.encode('utf-8'), diff[-len(expected):])

    def test_unshelve_args_delete_only(self):
        tree = self.make_branch_and_tree('tree')
        manager = tree.get_shelf_manager()
        shelf_file = manager.new_shelf()[1]
        try:
            shelf_file.write(b'garbage')
        finally:
            shelf_file.close()
        unshelver = shelf_ui.Unshelver.from_args(directory='tree', action='delete-only')
        try:
            unshelver.run()
        finally:
            unshelver.tree.unlock()
        self.assertIs(None, manager.last_shelf())

    def test_unshelve_args_invalid_shelf_id(self):
        tree = self.make_branch_and_tree('tree')
        manager = tree.get_shelf_manager()
        shelf_file = manager.new_shelf()[1]
        try:
            shelf_file.write(b'garbage')
        finally:
            shelf_file.close()
        self.assertRaises(shelf.InvalidShelfId, shelf_ui.Unshelver.from_args, directory='tree', action='delete-only', shelf_id='foo')