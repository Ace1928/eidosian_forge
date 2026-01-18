import os
from breezy import osutils, tests, workingtree
class TestJoin(tests.TestCaseWithTransport):

    def make_trees(self):
        base_tree = self.make_branch_and_tree('tree', format='development-subtree')
        base_tree.commit('empty commit')
        self.build_tree(['tree/subtree/', 'tree/subtree/file1'])
        sub_tree = self.make_branch_and_tree('tree/subtree')
        sub_tree.add('file1', ids=b'file1-id')
        sub_tree.commit('added file1')
        return (base_tree, sub_tree)

    def check_success(self, path):
        base_tree = workingtree.WorkingTree.open(path)
        self.assertEqual(b'file1-id', base_tree.path2id('subtree/file1'))

    def test_join(self):
        base_tree, sub_tree = self.make_trees()
        self.run_bzr('join tree/subtree')
        self.check_success('tree')

    def test_join_dot(self):
        base_tree, sub_tree = self.make_trees()
        self.run_bzr('join .', working_dir='tree/subtree')
        self.check_success('tree')

    def test_join_error(self):
        base_tree, sub_tree = self.make_trees()
        os.mkdir('tree/subtree2')
        osutils.rename('tree/subtree', 'tree/subtree2/subtree')
        self.run_bzr_error(('Cannot join .*subtree.  Parent directory is not versioned',), 'join tree/subtree2/subtree')
        self.run_bzr_error(('Not a branch:.*subtree2',), 'join tree/subtree2')

    def test_join_reference(self):
        """Join can add a reference if --reference is supplied"""
        base_tree, sub_tree = self.make_trees()
        subtree_root_id = sub_tree.path2id('')
        self.run_bzr('join . --reference', working_dir='tree/subtree')
        sub_tree.lock_read()
        self.addCleanup(sub_tree.unlock)
        if sub_tree.supports_setting_file_ids():
            self.assertEqual(b'file1-id', sub_tree.path2id('file1'))
            self.assertEqual('file1', sub_tree.id2path(b'file1-id'))
            self.assertEqual(subtree_root_id, sub_tree.path2id(''))
            self.assertEqual('', sub_tree.id2path(subtree_root_id))
            self.assertEqual(sub_tree.path2id('file1'), base_tree.path2id('subtree/file1'))
        base_tree.lock_read()
        self.addCleanup(base_tree.unlock)
        self.assertEqual(['subtree'], list(base_tree.iter_references()))
        if base_tree.supports_setting_file_ids():
            self.assertEqual(b'file1-id', sub_tree.path2id('file1'))
            self.assertEqual('file1', sub_tree.id2path(b'file1-id'))
            self.assertEqual(subtree_root_id, base_tree.path2id('subtree'))
            self.assertEqual('subtree', base_tree.id2path(subtree_root_id))

    def test_references_check_repository_support(self):
        """Users are stopped from adding a reference that can't be committed."""
        tree = self.make_branch_and_tree('tree', format='dirstate')
        tree2 = self.make_branch_and_tree('tree/subtree')
        out, err = self.run_bzr('join --reference tree/subtree', retcode=3)
        self.assertContainsRe(err, "Can't join trees")
        self.assertContainsRe(err, 'use brz upgrade')