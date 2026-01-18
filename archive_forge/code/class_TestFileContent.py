from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestFileContent(TestCaseWithTree):

    def test_get_file(self):
        work_tree = self.make_branch_and_tree('wt')
        tree = self.get_tree_no_parents_abc_content_2(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        file_without_path = tree.get_file('a')
        try:
            lines = file_without_path.readlines()
            self.assertEqual([b'foobar\n'], lines)
        finally:
            file_without_path.close()

    def test_get_file_context_manager(self):
        work_tree = self.make_branch_and_tree('wt')
        tree = self.get_tree_no_parents_abc_content_2(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        with tree.get_file('a') as f:
            self.assertEqual(b'foobar\n', f.read())

    def test_get_file_text(self):
        work_tree = self.make_branch_and_tree('wt')
        tree = self.get_tree_no_parents_abc_content_2(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(b'foobar\n', tree.get_file_text('a'))

    def test_get_file_lines(self):
        work_tree = self.make_branch_and_tree('wt')
        tree = self.get_tree_no_parents_abc_content_2(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual([b'foobar\n'], tree.get_file_lines('a'))

    def test_get_file_lines_multi_line_breaks(self):
        work_tree = self.make_branch_and_tree('wt')
        self.build_tree_contents([('wt/foobar', b'a\rb\nc\r\nd')])
        work_tree.add('foobar')
        tree = self._convert_tree(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual([b'a\rb\n', b'c\r\n', b'd'], tree.get_file_lines('foobar'))