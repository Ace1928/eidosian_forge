import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
class TestIterChanges(TestCaseWithTwoTrees):
    """Test the comparison iterator"""

    def _make_abc_tree(self, tree):
        """setup an abc content tree."""
        files = ['a', 'b/', 'b/c']
        self.build_tree(files, line_endings='binary', transport=tree.controldir.root_transport)
        tree.set_root_id(b'root-id')
        tree.add(files, ids=[b'a-id', b'b-id', b'c-id'])

    def get_tree_no_parents_abc_content(self, tree, converter=None):
        """return a test tree with a, b/, b/c contents."""
        self._make_abc_tree(tree)
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_7(self, tree, converter=None):
        """return a test tree with a, b/, d/e contents.

        This variation adds a dir 'd' (b'd-id'), renames b to d/e.
        """
        self._make_abc_tree(tree)
        self.build_tree(['d/'], transport=tree.controldir.root_transport)
        tree.add(['d'], ids=[b'd-id'])
        tt = tree.transform()
        trans_id = tt.trans_id_tree_path('b')
        parent_trans_id = tt.trans_id_tree_path('d')
        tt.adjust_path('e', parent_trans_id, trans_id)
        tt.apply()
        return self._convert_tree(tree, converter)

    def assertEqualIterChanges(self, left_changes, right_changes):
        """Assert that left_changes == right_changes.

        :param left_changes: A list of the output from iter_changes.
        :param right_changes: A list of the output from iter_changes.
        """
        left_changes = self.sorted(left_changes)
        right_changes = self.sorted(right_changes)
        if left_changes == right_changes:
            return
        left_dict = {item[0]: item for item in left_changes}
        right_dict = {item[0]: item for item in right_changes}
        if len(left_dict) != len(left_changes) or len(right_dict) != len(right_changes):
            self.assertEqual(left_changes, right_changes)
        keys = set(left_dict).union(set(right_dict))
        different = []
        same = []
        for key in keys:
            left_item = left_dict.get(key)
            right_item = right_dict.get(key)
            if left_item == right_item:
                same.append(str(left_item))
            else:
                different.append(' {}\n {}'.format(left_item, right_item))
        self.fail('iter_changes output different. Unchanged items:\n' + '\n'.join(same) + '\nChanged items:\n' + '\n'.join(different))

    def do_iter_changes(self, tree1, tree2, **extra_args):
        """Helper to run iter_changes from tree1 to tree2.

        :param tree1, tree2:  The source and target trees. These will be locked
            automatically.
        :param **extra_args: Extra args to pass to iter_changes. This is not
            inspected by this test helper.
        """
        with tree1.lock_read(), tree2.lock_read():
            return self.sorted(self.intertree_class(tree1, tree2).iter_changes(**extra_args))

    def check_has_changes(self, expected, tree1, tree2):
        if not isinstance(tree2, mutabletree.MutableTree):
            if isinstance(tree1, mutabletree.MutableTree):
                tree2, tree1 = (tree1, tree2)
            else:
                return
        with tree1.lock_read(), tree2.lock_read():
            return tree2.has_changes(tree1)

    def mutable_trees_to_locked_test_trees(self, tree1, tree2):
        """Convert the working trees into test trees.

        Read lock them, and add the unlock to the cleanup.
        """
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        tree1.lock_read()
        self.addCleanup(tree1.unlock)
        tree2.lock_read()
        self.addCleanup(tree2.unlock)
        return (tree1, tree2)

    def make_tree_with_special_names(self):
        """Create a tree with filenames chosen to exercise the walk order."""
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        paths = self._create_special_names(tree2, 'tree2')
        tree2.commit('initial', rev_id=b'rev-1')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        return (tree1, tree2, paths)

    def make_trees_with_special_names(self):
        """Both trees will use the special names.

        But the contents will differ for each file.
        """
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        paths = self._create_special_names(tree1, 'tree1')
        paths = self._create_special_names(tree2, 'tree2')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        return (tree1, tree2, paths)

    def _create_special_names(self, tree, base_path):
        """Create a tree with paths that expose differences in sort orders."""
        dirs = ['a', 'a-a', 'a/a', 'a/a-a', 'a/a/a', 'a/a/a-a', 'a/a/a/a', 'a/a/a/a-a', 'a/a/a/a/a']
        with_slashes = []
        paths = []
        path_ids = []
        for d in dirs:
            with_slashes.append(base_path + '/' + d + '/')
            with_slashes.append(base_path + '/' + d + '/f')
            paths.append(d)
            paths.append(d + '/f')
            path_ids.append((d.replace('/', '_') + '-id').encode('ascii'))
            path_ids.append((d.replace('/', '_') + '_f-id').encode('ascii'))
        self.build_tree(with_slashes)
        tree.add(paths, ids=path_ids)
        return paths

    def test_compare_empty_trees(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_no_content(tree1)
        tree2 = self.get_tree_no_parents_no_content(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        self.assertEqual([], self.do_iter_changes(tree1, tree2))
        self.check_has_changes(False, tree1, tree2)

    def added(self, tree, path):
        entry = self.get_path_entry(tree, path)
        return InventoryTreeChange(entry.file_id, (None, path), True, (False, True), (None, entry.parent_id), (None, entry.name), (None, entry.kind), (None, entry.executable))

    @staticmethod
    def get_path_entry(tree, path):
        iterator = tree.iter_entries_by_dir(specific_files=[path])
        try:
            return next(iterator)[1]
        except StopIteration:
            raise KeyError(path)

    def changed_content(self, tree, path):
        entry = self.get_path_entry(tree, path)
        return InventoryTreeChange(entry.file_id, (path, path), True, (True, True), (entry.parent_id, entry.parent_id), (entry.name, entry.name), (entry.kind, entry.kind), (entry.executable, entry.executable))

    def kind_changed(self, from_tree, to_tree, from_path, to_path):
        old_entry = self.get_path_entry(from_tree, from_path)
        new_entry = self.get_path_entry(to_tree, to_path)
        return InventoryTreeChange(new_entry.file_id, (from_path, to_path), True, (True, True), (old_entry.parent_id, new_entry.parent_id), (old_entry.name, new_entry.name), (old_entry.kind, new_entry.kind), (old_entry.executable, new_entry.executable))

    def missing(self, file_id, from_path, to_path, parent_id, kind):
        _, from_basename = os.path.split(from_path)
        _, to_basename = os.path.split(to_path)
        return InventoryTreeChange(file_id, (from_path, to_path), True, (True, True), (parent_id, parent_id), (from_basename, to_basename), (kind, None), (False, False))

    def deleted(self, tree, path):
        entry = self.get_path_entry(tree, path)
        return InventoryTreeChange(entry.file_id, (path, None), True, (True, False), (entry.parent_id, None), (entry.name, None), (entry.kind, None), (entry.executable, None))

    def renamed(self, from_tree, to_tree, from_path, to_path, content_changed):
        from_entry = self.get_path_entry(from_tree, from_path)
        to_entry = self.get_path_entry(to_tree, to_path)
        return InventoryTreeChange(to_entry.file_id, (from_path, to_path), content_changed, (True, True), (from_entry.parent_id, to_entry.parent_id), (from_entry.name, to_entry.name), (from_entry.kind, to_entry.kind), (from_entry.executable, to_entry.executable))

    def unchanged(self, tree, path):
        entry = self.get_path_entry(tree, path)
        parent = entry.parent_id
        name = entry.name
        kind = entry.kind
        executable = entry.executable
        return InventoryTreeChange(entry.file_id, (path, path), False, (True, True), (parent, parent), (name, name), (kind, kind), (executable, executable))

    def unversioned(self, tree, path):
        """Create an unversioned result."""
        _, basename = os.path.split(path)
        kind = tree._comparison_data(None, path)[0]
        return InventoryTreeChange(None, (None, path), True, (False, False), (None, None), (None, basename), (None, kind), (None, False))

    def sorted(self, changes):
        return sorted(changes, key=_change_key)

    def test_empty_to_abc_content(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_no_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected_results = self.sorted([self.added(tree2, ''), self.added(tree2, 'a'), self.added(tree2, 'b'), self.added(tree2, 'b/c'), self.deleted(tree1, '')])
        self.assertEqual(expected_results, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_empty_specific_files(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_no_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([], self.do_iter_changes(tree1, tree2, specific_files=[]))

    def test_no_specific_files(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_no_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected_results = self.sorted([self.added(tree2, ''), self.added(tree2, 'a'), self.added(tree2, 'b'), self.added(tree2, 'b/c'), self.deleted(tree1, '')])
        self.assertEqual(expected_results, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_empty_to_abc_content_a_only(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_no_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual(self.sorted([self.added(tree2, ''), self.added(tree2, 'a'), self.deleted(tree1, '')]), self.do_iter_changes(tree1, tree2, specific_files=['a']))

    def test_abc_content_to_empty_a_only(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_no_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.deleted(tree1, 'a')], self.do_iter_changes(tree1, tree2, specific_files=['a']))

    def test_abc_content_to_empty_b_only(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_no_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.deleted(tree1, 'b'), self.deleted(tree1, 'b/c')], self.do_iter_changes(tree1, tree2, specific_files=['b']))

    def test_empty_to_abc_content_a_and_c_only(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_no_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected_result = self.sorted([self.added(tree2, ''), self.added(tree2, 'a'), self.added(tree2, 'b'), self.added(tree2, 'b/c'), self.deleted(tree1, '')])
        self.assertEqual(expected_result, self.do_iter_changes(tree1, tree2, specific_files=['a', 'b/c']))

    def test_abc_content_to_empty(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_no_content(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected_results = self.sorted([self.added(tree2, ''), self.deleted(tree1, ''), self.deleted(tree1, 'a'), self.deleted(tree1, 'b'), self.deleted(tree1, 'b/c')])
        self.assertEqual(expected_results, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_content_modification(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_2(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqual([(b'a-id', ('a', 'a'), True, (True, True), (root_id, root_id), ('a', 'a'), ('file', 'file'), (False, False), False)], self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_meta_modification(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_3(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        self.assertEqual([(b'c-id', ('b/c', 'b/c'), False, (True, True), (b'b-id', b'b-id'), ('c', 'c'), ('file', 'file'), (False, True), False)], self.do_iter_changes(tree1, tree2))

    def test_empty_dir(self):
        """an empty dir should not cause glitches to surrounding files."""
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        self.build_tree(['1/a-empty/', '2/a-empty/'])
        tree1.add(['a-empty'], ids=[b'a-empty'])
        tree2.add(['a-empty'], ids=[b'a-empty'])
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        expected = []
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))

    def test_file_rename(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_4(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqual([(tree1.path2id('a'), ('a', 'd'), False, (True, True), (root_id, root_id), ('a', 'd'), ('file', 'file'), (False, False), False)], self.do_iter_changes(tree1, tree2))

    def test_file_rename_and_modification(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_5(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqual([(b'a-id', ('a', 'd'), True, (True, True), (root_id, root_id), ('a', 'd'), ('file', 'file'), (False, False), False)], self.do_iter_changes(tree1, tree2))

    def test_specific_content_modification_grabs_parents(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.mkdir('changing', b'parent-id')
        tree1.mkdir('changing/unchanging', b'mid-id')
        tree1.add(['changing/unchanging/file'], ['file'], [b'file-id'])
        tree1.put_file_bytes_non_atomic('changing/unchanging/file', b'a file')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(tree1.path2id(''))
        tree2.mkdir('changed', b'parent-id')
        tree2.mkdir('changed/unchanging', b'mid-id')
        tree2.add(['changed/unchanging/file'], ['file'], [b'file-id'])
        tree2.put_file_bytes_non_atomic('changed/unchanging/file', b'changed content')
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqualIterChanges([self.renamed(tree1, tree2, 'changing', 'changed', False), self.renamed(tree1, tree2, 'changing/unchanging/file', 'changed/unchanging/file', True)], self.do_iter_changes(tree1, tree2, specific_files=['changed/unchanging/file']))

    def test_specific_content_modification_grabs_parents_root_changes(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.set_root_id(b'old')
        tree1.mkdir('changed', b'parent-id')
        tree1.mkdir('changed/unchanging', b'mid-id')
        tree1.add(['changed/unchanging/file'], ['file'], [b'file-id'])
        tree1.put_file_bytes_non_atomic('changed/unchanging/file', b'a file')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(b'new')
        tree2.mkdir('changed', b'parent-id')
        tree2.mkdir('changed/unchanging', b'mid-id')
        tree2.add(['changed/unchanging/file'], ['file'], [b'file-id'])
        tree2.put_file_bytes_non_atomic('changed/unchanging/file', b'changed content')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqualIterChanges([self.renamed(tree1, tree2, 'changed', 'changed', False), self.added(tree2, ''), self.deleted(tree1, ''), self.renamed(tree1, tree2, 'changed/unchanging/file', 'changed/unchanging/file', True)], self.do_iter_changes(tree1, tree2, specific_files=['changed/unchanging/file']))

    def test_specific_with_rename_under_new_dir_reports_new_dir(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_7(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqualIterChanges([self.renamed(tree1, tree2, 'b', 'd/e', False), self.added(tree2, 'd')], self.do_iter_changes(tree1, tree2, specific_files=['d/e']))

    def test_specific_with_rename_under_dir_under_new_dir_reports_new_dir(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_7(tree2)
        tree2.rename_one('a', 'd/e/a')
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqualIterChanges([self.renamed(tree1, tree2, 'b', 'd/e', False), self.added(tree2, 'd'), self.renamed(tree1, tree2, 'a', 'd/e/a', False)], self.do_iter_changes(tree1, tree2, specific_files=['d/e/a']))

    def test_specific_old_parent_same_path_new_parent(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.add(['a'], ['file'], [b'a-id'])
        tree1.put_file_bytes_non_atomic('a', b'a file')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(tree1.path2id(''))
        tree2.mkdir('a', b'b-id')
        tree2.add(['a/c'], ['file'], [b'c-id'])
        tree2.put_file_bytes_non_atomic('a/c', b'another file')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqualIterChanges([self.deleted(tree1, 'a'), self.added(tree2, 'a'), self.added(tree2, 'a/c')], self.do_iter_changes(tree1, tree2, specific_files=['a/c']))

    def test_specific_old_parent_becomes_file(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.mkdir('a', b'a-old-id')
        tree1.mkdir('a/reparented', b'reparented-id')
        tree1.mkdir('a/deleted', b'deleted-id')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(tree1.path2id(''))
        tree2.mkdir('a', b'a-new-id')
        tree2.mkdir('a/reparented', b'reparented-id')
        tree2.add(['b'], ['file'], [b'a-old-id'])
        tree2.put_file_bytes_non_atomic('b', b'')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqualIterChanges([self.kind_changed(tree1, tree2, 'a', 'b'), self.added(tree2, 'a'), self.renamed(tree1, tree2, 'a/reparented', 'a/reparented', False), self.deleted(tree1, 'a/deleted')], self.do_iter_changes(tree1, tree2, specific_files=['a/reparented']))

    def test_specific_old_parent_is_deleted(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.mkdir('a', b'a-old-id')
        tree1.mkdir('a/reparented', b'reparented-id')
        tree1.mkdir('a/deleted', b'deleted-id')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(tree1.path2id(''))
        tree2.mkdir('a', b'a-new-id')
        tree2.mkdir('a/reparented', b'reparented-id')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqualIterChanges([self.deleted(tree1, 'a'), self.added(tree2, 'a'), self.renamed(tree1, tree2, 'a/reparented', 'a/reparented', False), self.deleted(tree1, 'a/deleted')], self.do_iter_changes(tree1, tree2, specific_files=['a/reparented']))

    def test_specific_old_parent_child_collides_with_unselected_new(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.mkdir('a', b'a-old-id')
        tree1.mkdir('a/reparented', b'reparented-id')
        tree1.mkdir('collides', b'collides-id')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(tree1.path2id(''))
        tree2.mkdir('a', b'a-new-id')
        tree2.mkdir('a/selected', b'selected-id')
        tree2.mkdir('collides', b'reparented-id')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqualIterChanges([self.deleted(tree1, 'a'), self.added(tree2, 'a'), self.renamed(tree1, tree2, 'a/reparented', 'collides', False), self.deleted(tree1, 'collides'), self.added(tree2, 'a/selected')], self.do_iter_changes(tree1, tree2, specific_files=['a/selected']))

    def test_specific_old_parent_child_dir_stops_being_dir(self):
        tree1 = self.make_branch_and_tree('1')
        tree1.mkdir('a', b'a-old-id')
        tree1.mkdir('a/reparented', b'reparented-id-1')
        tree1.mkdir('a/deleted', b'deleted-id-1')
        tree1.mkdir('a/deleted/reparented', b'reparented-id-2')
        tree1.mkdir('a/deleted/deleted', b'deleted-id-2')
        tree2 = self.make_to_branch_and_tree('2')
        tree2.set_root_id(tree1.path2id(''))
        tree2.mkdir('a', b'a-new-id')
        tree2.mkdir('a/reparented', b'reparented-id-1')
        tree2.mkdir('reparented', b'reparented-id-2')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqualIterChanges([self.deleted(tree1, 'a'), self.added(tree2, 'a'), self.renamed(tree1, tree2, 'a/reparented', 'a/reparented', False), self.renamed(tree1, tree2, 'a/deleted/reparented', 'reparented', False), self.deleted(tree1, 'a/deleted'), self.deleted(tree1, 'a/deleted/deleted')], self.do_iter_changes(tree1, tree2, specific_files=['a/reparented']))

    def test_file_rename_and_meta_modification(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_6(tree2)
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        root_id = tree1.path2id('')
        self.assertEqual([(b'c-id', ('b/c', 'e'), False, (True, True), (b'b-id', root_id), ('c', 'e'), ('file', 'file'), (False, True), False)], self.do_iter_changes(tree1, tree2))

    def test_file_becomes_unversionable_bug_438569(self):
        self.requireFeature(features.OsFifoFeature)
        tree1 = self.make_branch_and_tree('1')
        self.build_tree(['1/a'])
        tree1.set_root_id(b'root-id')
        tree1.add(['a'], ids=[b'a-id'])
        tree2 = self.make_branch_and_tree('2')
        os.mkfifo('2/a')
        tree2.add(['a'], ['file'], [b'a-id'])
        try:
            tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        except (KeyError,):
            raise tests.TestNotApplicable('Cannot represent a FIFO in this case %s' % self.id())
        try:
            self.do_iter_changes(tree1, tree2)
        except errors.BadFileKindError:
            pass

    def test_missing_in_target(self):
        """Test with the target files versioned but absent from disk."""
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content(tree2)
        os.unlink('2/a')
        shutil.rmtree('2/b')
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        self.not_applicable_if_missing_in('a', tree2)
        self.not_applicable_if_missing_in('b', tree2)
        expected = self.sorted([self.missing(b'a-id', 'a', 'a', b'root-id', 'file'), self.missing(b'b-id', 'b', 'b', b'root-id', 'directory'), self.missing(b'c-id', 'b/c', 'b/c', b'b-id', 'file')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))

    def test_missing_and_renamed(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree1/file'])
        tree1.add(['file'], ids=[b'file-id'])
        self.build_tree(['tree2/directory/'])
        tree2.add(['directory'], ids=[b'file-id'])
        os.rmdir('tree2/directory')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_missing_in('directory', tree2)
        root_id = tree1.path2id('')
        expected = self.sorted([self.missing(b'file-id', 'file', 'directory', root_id, 'file')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))

    def test_only_in_source_and_missing(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree1/file'])
        tree1.add(['file'], ids=[b'file-id'])
        os.unlink('tree1/file')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_missing_in('file', tree1)
        root_id = tree1.path2id('')
        expected = [InventoryTreeChange(b'file-id', ('file', None), False, (True, False), (root_id, None), ('file', None), (None, None), (False, None))]
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))

    def test_only_in_target_and_missing(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree2/file'])
        tree2.add(['file'], ids=[b'file-id'])
        os.unlink('tree2/file')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_missing_in('file', tree2)
        root_id = tree1.path2id('')
        expected = [InventoryTreeChange(b'file-id', (None, 'file'), False, (False, True), (None, root_id), (None, 'file'), (None, None), (None, False))]
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))

    def test_only_in_target_missing_subtree_specific_bug_367632(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree2/a-dir/', 'tree2/a-dir/a-file'])
        tree2.add(['a-dir', 'a-dir/a-file'], ids=[b'dir-id', b'file-id'])
        os.unlink('tree2/a-dir/a-file')
        os.rmdir('tree2/a-dir')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_missing_in('a-dir', tree2)
        root_id = tree1.path2id('')
        expected = [InventoryTreeChange(b'dir-id', (None, 'a-dir'), False, (False, True), (None, root_id), (None, 'a-dir'), (None, None), (None, False)), InventoryTreeChange(b'file-id', (None, 'a-dir/a-file'), False, (False, True), (None, b'dir-id'), (None, 'a-file'), (None, None), (None, False))]
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['']))

    def test_unchanged_with_renames_and_modifications(self):
        """want_unchanged should generate a list of unchanged entries."""
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        tree1 = self.get_tree_no_parents_abc_content(tree1)
        tree2 = self.get_tree_no_parents_abc_content_5(tree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual(sorted([self.unchanged(tree1, ''), self.unchanged(tree1, 'b'), InventoryTreeChange(b'a-id', ('a', 'd'), True, (True, True), (b'root-id', b'root-id'), ('a', 'd'), ('file', 'file'), (False, False)), self.unchanged(tree1, 'b/c')]), self.do_iter_changes(tree1, tree2, include_unchanged=True))

    def test_compare_subtrees(self):
        tree1 = self.make_branch_and_tree('1')
        if not tree1.supports_tree_reference():
            return
        tree1.set_root_id(b'root-id')
        subtree1 = self.make_branch_and_tree('1/sub')
        subtree1.set_root_id(b'subtree-id')
        tree1.add_reference(subtree1)
        tree2 = self.make_to_branch_and_tree('2')
        if not tree2.supports_tree_reference():
            return
        tree2.set_root_id(b'root-id')
        subtree2 = self.make_to_branch_and_tree('2/sub')
        subtree2.set_root_id(b'subtree-id')
        tree2.add_reference(subtree2)
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([], list(tree2.iter_changes(tree1)))
        subtree1.commit('commit', rev_id=b'commit-a')
        self.assertThat(tree2.iter_changes(tree1, include_unchanged=True), MatchesTreeChanges(tree1, tree2, [TreeChange(('', ''), False, (True, True), ('', ''), ('directory', 'directory'), (False, False)), TreeChange(('sub', 'sub'), False, (True, True), ('sub', 'sub'), ('tree-reference', 'tree-reference'), (False, False))]))

    def test_disk_in_subtrees_skipped(self):
        """subtrees are considered not-in-the-current-tree.

        This test tests the trivial case, where the basis has no paths in the
        current trees subtree.
        """
        tree1 = self.make_branch_and_tree('1')
        tree1.set_root_id(b'root-id')
        tree2 = self.make_to_branch_and_tree('2')
        if not tree2.supports_tree_reference():
            return
        tree2.set_root_id(b'root-id')
        subtree2 = self.make_to_branch_and_tree('2/sub')
        subtree2.set_root_id(b'subtree-id')
        tree2.add_reference(subtree2)
        self.build_tree(['2/sub/file'])
        subtree2.add(['file'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.added(tree2, 'sub')], self.do_iter_changes(tree1, tree2, want_unversioned=True))
        self.assertEqual([self.added(tree2, 'sub')], self.do_iter_changes(tree1, tree2, specific_files=['sub'], want_unversioned=True))

    def test_default_ignores_unversioned_files(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree1/a', 'tree1/c', 'tree2/a', 'tree2/b', 'tree2/c'])
        tree1.add(['a', 'c'], ids=[b'a-id', b'c-id'])
        tree2.add(['a', 'c'], ids=[b'a-id', b'c-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected = self.sorted([self.changed_content(tree2, 'a'), self.changed_content(tree2, 'c')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_unversioned_paths_in_tree(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree2/file', 'tree2/dir/'])
        if supports_symlinks(self.test_dir):
            os.symlink('target', 'tree2/link')
            links_supported = True
        else:
            links_supported = False
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        expected = [self.unversioned(tree2, 'file'), self.unversioned(tree2, 'dir')]
        if links_supported:
            expected.append(self.unversioned(tree2, 'link'))
        expected = self.sorted(expected)
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))

    def test_unversioned_paths_in_tree_specific_files(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        self.build_tree(['tree2/file', 'tree2/dir/'])
        if supports_symlinks(self.test_dir):
            os.symlink('target', 'tree2/link')
            links_supported = True
        else:
            links_supported = False
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        expected = [self.unversioned(tree2, 'file'), self.unversioned(tree2, 'dir')]
        specific_files = ['file', 'dir']
        if links_supported:
            expected.append(self.unversioned(tree2, 'link'))
            specific_files.append('link')
        expected = self.sorted(expected)
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=specific_files, require_versioned=False, want_unversioned=True))

    def test_unversioned_paths_in_target_matching_source_old_names(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree2/file', 'tree2/dir/', 'tree1/file', 'tree2/movedfile', 'tree1/dir/', 'tree2/moveddir/'])
        if supports_symlinks(self.test_dir):
            os.symlink('target', 'tree1/link')
            os.symlink('target', 'tree2/link')
            os.symlink('target', 'tree2/movedlink')
            links_supported = True
        else:
            links_supported = False
        tree1.add(['file', 'dir'], ids=[b'file-id', b'dir-id'])
        tree2.add(['movedfile', 'moveddir'], ids=[b'file-id', b'dir-id'])
        if links_supported:
            tree1.add(['link'], ids=[b'link-id'])
            tree2.add(['movedlink'], ids=[b'link-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        root_id = tree1.path2id('')
        expected = [self.renamed(tree1, tree2, 'dir', 'moveddir', False), self.renamed(tree1, tree2, 'file', 'movedfile', True), self.unversioned(tree2, 'file'), self.unversioned(tree2, 'dir')]
        specific_files = ['file', 'dir']
        if links_supported:
            expected.append(self.renamed(tree1, tree2, 'link', 'movedlink', False))
            expected.append(self.unversioned(tree2, 'link'))
            specific_files.append('link')
        expected = self.sorted(expected)
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=specific_files, require_versioned=False, want_unversioned=True))

    def test_similar_filenames(self):
        """Test when we have a few files with similar names."""
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree1/a/', 'tree1/a/b/', 'tree1/a/b/c/', 'tree1/a/b/c/d/', 'tree1/a-c/', 'tree1/a-c/e/', 'tree2/a/', 'tree2/a/b/', 'tree2/a/b/c/', 'tree2/a/b/c/d/', 'tree2/a-c/', 'tree2/a-c/e/'])
        tree1.add(['a', 'a/b', 'a/b/c', 'a/b/c/d', 'a-c', 'a-c/e'], ids=[b'a-id', b'b-id', b'c-id', b'd-id', b'a-c-id', b'e-id'])
        tree2.add(['a', 'a/b', 'a/b/c', 'a/b/c/d', 'a-c', 'a-c/e'], ids=[b'a-id', b'b-id', b'c-id', b'd-id', b'a-c-id', b'e-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        self.assertEqual([], self.do_iter_changes(tree1, tree2, want_unversioned=True))
        expected = self.sorted([self.unchanged(tree2, ''), self.unchanged(tree2, 'a'), self.unchanged(tree2, 'a/b'), self.unchanged(tree2, 'a/b/c'), self.unchanged(tree2, 'a/b/c/d'), self.unchanged(tree2, 'a-c'), self.unchanged(tree2, 'a-c/e')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True, include_unchanged=True))

    def test_unversioned_subtree_only_emits_root(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree2/dir/', 'tree2/dir/file'])
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        expected = [self.unversioned(tree2, 'dir')]
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))

    def make_trees_with_symlinks(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree1/fromfile', 'tree1/fromdir/'])
        self.build_tree(['tree2/tofile', 'tree2/todir/', 'tree2/unknown'])
        os.symlink('original', 'tree1/changed')
        os.symlink('original', 'tree1/removed')
        os.symlink('original', 'tree1/tofile')
        os.symlink('original', 'tree1/todir')
        os.symlink('unknown', 'tree1/unchanged')
        os.symlink('new', 'tree2/added')
        os.symlink('new', 'tree2/changed')
        os.symlink('new', 'tree2/fromfile')
        os.symlink('new', 'tree2/fromdir')
        os.symlink('unknown', 'tree2/unchanged')
        from_paths_and_ids = ['fromdir', 'fromfile', 'changed', 'removed', 'todir', 'tofile', 'unchanged']
        to_paths_and_ids = ['added', 'fromdir', 'fromfile', 'changed', 'todir', 'tofile', 'unchanged']
        tree1.add(from_paths_and_ids, ids=[p.encode('utf-8') for p in from_paths_and_ids])
        tree2.add(to_paths_and_ids, ids=[p.encode('utf-8') for p in to_paths_and_ids])
        return self.mutable_trees_to_locked_test_trees(tree1, tree2)

    def test_versioned_symlinks(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tree1, tree2 = self.make_trees_with_symlinks()
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        root_id = tree1.path2id('')
        expected = [self.unchanged(tree1, ''), self.added(tree2, 'added'), self.changed_content(tree2, 'changed'), self.kind_changed(tree1, tree2, 'fromdir', 'fromdir'), self.kind_changed(tree1, tree2, 'fromfile', 'fromfile'), self.deleted(tree1, 'removed'), self.unchanged(tree2, 'unchanged'), self.unversioned(tree2, 'unknown'), self.kind_changed(tree1, tree2, 'todir', 'todir'), self.kind_changed(tree1, tree2, 'tofile', 'tofile')]
        expected = self.sorted(expected)
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, include_unchanged=True, want_unversioned=True))
        self.check_has_changes(True, tree1, tree2)

    def test_versioned_symlinks_specific_files(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tree1, tree2 = self.make_trees_with_symlinks()
        root_id = tree1.path2id('')
        expected = [self.added(tree2, 'added'), self.changed_content(tree2, 'changed'), self.kind_changed(tree1, tree2, 'fromdir', 'fromdir'), self.kind_changed(tree1, tree2, 'fromfile', 'fromfile'), self.deleted(tree1, 'removed'), self.kind_changed(tree1, tree2, 'todir', 'todir'), self.kind_changed(tree1, tree2, 'tofile', 'tofile')]
        expected = self.sorted(expected)
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['added', 'changed', 'fromdir', 'fromfile', 'removed', 'unchanged', 'todir', 'tofile']))
        self.check_has_changes(True, tree1, tree2)

    def test_tree_with_special_names(self):
        tree1, tree2, paths = self.make_tree_with_special_names()
        expected = self.sorted((self.added(tree2, p) for p in paths))
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_trees_with_special_names(self):
        tree1, tree2, paths = self.make_trees_with_special_names()
        expected = self.sorted((self.changed_content(tree2, p) for p in paths if p.endswith('/f')))
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_trees_with_deleted_dir(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        self.build_tree(['tree1/a', 'tree1/b/', 'tree1/b/c', 'tree1/b/d/', 'tree1/b/d/e', 'tree1/f/', 'tree1/f/g', 'tree2/a', 'tree2/f/', 'tree2/f/g'])
        tree1.add(['a', 'b', 'b/c', 'b/d/', 'b/d/e', 'f', 'f/g'], ids=[b'a-id', b'b-id', b'c-id', b'd-id', b'e-id', b'f-id', b'g-id'])
        tree2.add(['a', 'f', 'f/g'], ids=[b'a-id', b'f-id', b'g-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected = [self.changed_content(tree2, 'a'), self.changed_content(tree2, 'f/g'), self.deleted(tree1, 'b'), self.deleted(tree1, 'b/c'), self.deleted(tree1, 'b/d'), self.deleted(tree1, 'b/d/e')]
        self.assertEqualIterChanges(expected, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_added_unicode(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        a_id = 'α-id'.encode()
        added_id = 'ω_added_id'.encode()
        added_path = 'α/ω-added'
        try:
            self.build_tree(['tree1/α/', 'tree2/α/', 'tree2/α/ω-added'])
        except UnicodeError:
            raise tests.TestSkipped('Could not create Unicode files.')
        tree1.add(['α'], ids=[a_id])
        tree2.add(['α', 'α/ω-added'], ids=[a_id, added_id])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.added(tree2, added_path)], self.do_iter_changes(tree1, tree2))
        self.assertEqual([self.added(tree2, added_path)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
        self.check_has_changes(True, tree1, tree2)

    def test_deleted_unicode(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        a_id = 'α-id'.encode()
        deleted_id = 'ω_deleted_id'.encode()
        deleted_path = 'α/ω-deleted'
        try:
            self.build_tree(['tree1/α/', 'tree1/α/ω-deleted', 'tree2/α/'])
        except UnicodeError:
            raise tests.TestSkipped('Could not create Unicode files.')
        tree1.add(['α', 'α/ω-deleted'], ids=[a_id, deleted_id])
        tree2.add(['α'], ids=[a_id])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.deleted(tree1, deleted_path)], self.do_iter_changes(tree1, tree2))
        self.assertEqual([self.deleted(tree1, deleted_path)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
        self.check_has_changes(True, tree1, tree2)

    def test_modified_unicode(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        a_id = 'α-id'.encode()
        mod_id = 'ω_mod_id'.encode()
        mod_path = 'α/ω-modified'
        try:
            self.build_tree(['tree1/α/', 'tree1/' + mod_path, 'tree2/α/', 'tree2/' + mod_path])
        except UnicodeError:
            raise tests.TestSkipped('Could not create Unicode files.')
        tree1.add(['α', mod_path], ids=[a_id, mod_id])
        tree2.add(['α', mod_path], ids=[a_id, mod_id])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.changed_content(tree1, mod_path)], self.do_iter_changes(tree1, tree2))
        self.assertEqual([self.changed_content(tree1, mod_path)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
        self.check_has_changes(True, tree1, tree2)

    def test_renamed_unicode(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        a_id = 'α-id'.encode()
        rename_id = 'ω_rename_id'.encode()
        try:
            self.build_tree(['tree1/α/', 'tree2/α/'])
        except UnicodeError:
            raise tests.TestSkipped('Could not create Unicode files.')
        self.build_tree_contents([('tree1/ω-source', b'contents\n'), ('tree2/α/ω-target', b'contents\n')])
        tree1.add(['α', 'ω-source'], ids=[a_id, rename_id])
        tree2.add(['α', 'α/ω-target'], ids=[a_id, rename_id])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.assertEqual([self.renamed(tree1, tree2, 'ω-source', 'α/ω-target', False)], self.do_iter_changes(tree1, tree2))
        self.assertEqualIterChanges([self.renamed(tree1, tree2, 'ω-source', 'α/ω-target', False)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
        self.check_has_changes(True, tree1, tree2)

    def test_unchanged_unicode(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        a_id = 'α-id'.encode()
        subfile_id = 'ω-subfile-id'.encode()
        rootfile_id = 'ω-root-id'.encode()
        try:
            self.build_tree(['tree1/α/', 'tree2/α/'])
        except UnicodeError:
            raise tests.TestSkipped('Could not create Unicode files.')
        self.build_tree_contents([('tree1/α/ω-subfile', b'sub contents\n'), ('tree2/α/ω-subfile', b'sub contents\n'), ('tree1/ω-rootfile', b'root contents\n'), ('tree2/ω-rootfile', b'root contents\n')])
        tree1.add(['α', 'α/ω-subfile', 'ω-rootfile'], ids=[a_id, subfile_id, rootfile_id])
        tree2.add(['α', 'α/ω-subfile', 'ω-rootfile'], ids=[a_id, subfile_id, rootfile_id])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected = self.sorted([self.unchanged(tree1, ''), self.unchanged(tree1, 'α'), self.unchanged(tree1, 'α/ω-subfile'), self.unchanged(tree1, 'ω-rootfile')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, include_unchanged=True))
        expected = self.sorted([self.unchanged(tree1, 'α'), self.unchanged(tree1, 'α/ω-subfile')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['α'], include_unchanged=True))

    def test_unknown_unicode(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        tree2.set_root_id(tree1.path2id(''))
        a_id = 'α-id'.encode()
        try:
            self.build_tree(['tree1/α/', 'tree2/α/', 'tree2/α/unknown_dir/', 'tree2/α/unknown_file', 'tree2/α/unknown_dir/file', 'tree2/ω-unknown_root_file'])
        except UnicodeError:
            raise tests.TestSkipped('Could not create Unicode files.')
        tree1.add(['α'], ids=[a_id])
        tree2.add(['α'], ids=[a_id])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        expected = self.sorted([self.unversioned(tree2, 'α/unknown_dir'), self.unversioned(tree2, 'α/unknown_file'), self.unversioned(tree2, 'ω-unknown_root_file')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))
        self.assertEqual([], self.do_iter_changes(tree1, tree2))
        self.check_has_changes(False, tree1, tree2)
        expected = self.sorted([self.unversioned(tree2, 'α/unknown_dir'), self.unversioned(tree2, 'α/unknown_file')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['α'], require_versioned=False, want_unversioned=True))
        self.assertEqual([], self.do_iter_changes(tree1, tree2, specific_files=['α']))

    def test_unknown_empty_dir(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        self.build_tree(['tree1/a/', 'tree1/b/', 'tree2/a/', 'tree2/b/'])
        self.build_tree_contents([('tree1/b/file', b'contents\n'), ('tree2/b/file', b'contents\n')])
        tree1.add(['a', 'b', 'b/file'], ids=[b'a-id', b'b-id', b'b-file-id'])
        tree2.add(['a', 'b', 'b/file'], ids=[b'a-id', b'b-id', b'b-file-id'])
        self.build_tree(['tree2/a/file', 'tree2/a/dir/', 'tree2/a/dir/subfile'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        if tree2.has_versioned_directories():
            expected = self.sorted([self.unversioned(tree2, 'a/file'), self.unversioned(tree2, 'a/dir')])
            self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))
        else:
            expected = self.sorted([self.unversioned(tree2, 'a/file'), self.unversioned(tree2, 'a/dir/subfile')])
            self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))

    def test_rename_over_deleted(self):
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        self.build_tree_contents([('tree1/a', b'a contents\n'), ('tree1/b', b'b contents\n'), ('tree1/c', b'c contents\n'), ('tree1/d', b'd contents\n'), ('tree2/a', b'b contents\n'), ('tree2/d', b'c contents\n')])
        tree1.add(['a', 'b', 'c', 'd'], ids=[b'a-id', b'b-id', b'c-id', b'd-id'])
        tree2.add(['a', 'd'], ids=[b'b-id', b'c-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected = self.sorted([self.deleted(tree1, 'a'), self.deleted(tree1, 'd'), self.renamed(tree1, tree2, 'b', 'a', False), self.renamed(tree1, tree2, 'c', 'd', False)])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
        self.check_has_changes(True, tree1, tree2)

    def test_deleted_and_unknown(self):
        """Test a file marked removed, but still present on disk."""
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        self.build_tree_contents([('tree1/a', b'a contents\n'), ('tree1/b', b'b contents\n'), ('tree1/c', b'c contents\n'), ('tree2/a', b'a contents\n'), ('tree2/b', b'b contents\n'), ('tree2/c', b'c contents\n')])
        tree1.add(['a', 'b', 'c'], ids=[b'a-id', b'b-id', b'c-id'])
        tree2.add(['a', 'c'], ids=[b'a-id', b'c-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_cannot_represent_unversioned(tree2)
        expected = self.sorted([self.deleted(tree1, 'b'), self.unversioned(tree2, 'b')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))
        expected = self.sorted([self.deleted(tree1, 'b')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=False))

    def test_renamed_and_added(self):
        """Test when we have renamed a file, and put another in its place."""
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        self.build_tree_contents([('tree1/b', b'b contents\n'), ('tree1/c', b'c contents\n'), ('tree2/a', b'b contents\n'), ('tree2/b', b'new b contents\n'), ('tree2/c', b'new c contents\n'), ('tree2/d', b'c contents\n')])
        tree1.add(['b', 'c'], ids=[b'b1-id', b'c1-id'])
        tree2.add(['a', 'b', 'c', 'd'], ids=[b'b1-id', b'b2-id', b'c2-id', b'c1-id'])
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        expected = self.sorted([self.renamed(tree1, tree2, 'b', 'a', False), self.renamed(tree1, tree2, 'c', 'd', False), self.added(tree2, 'b'), self.added(tree2, 'c')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))

    def test_renamed_and_unknown(self):
        """A file was moved on the filesystem, but not in bzr."""
        tree1 = self.make_branch_and_tree('tree1')
        tree2 = self.make_to_branch_and_tree('tree2')
        root_id = tree1.path2id('')
        tree2.set_root_id(root_id)
        self.build_tree_contents([('tree1/a', b'a contents\n'), ('tree1/b', b'b contents\n'), ('tree2/a', b'a contents\n'), ('tree2/b', b'b contents\n')])
        tree1.add(['a', 'b'], ids=[b'a-id', b'b-id'])
        tree2.add(['a', 'b'], ids=[b'a-id', b'b-id'])
        os.rename('tree2/a', 'tree2/a2')
        tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
        self.not_applicable_if_missing_in('a', tree2)
        expected = self.sorted([self.missing(b'a-id', 'a', 'a', tree2.path2id(''), 'file'), self.unversioned(tree2, 'a2')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))