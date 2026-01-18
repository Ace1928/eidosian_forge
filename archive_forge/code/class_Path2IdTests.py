from breezy import errors, tests
from breezy.tests.per_tree import TestCaseWithTree
class Path2IdTests(TestCaseWithTree):

    def setUp(self):
        super().setUp()
        work_a = self.make_branch_and_tree('wta')
        if not work_a.supports_setting_file_ids():
            self.skipTest('working tree does not support setting file ids')
        self.build_tree(['wta/bla', 'wta/dir/', 'wta/dir/file'])
        work_a.add(['bla', 'dir', 'dir/file'], ids=[b'bla-id', b'dir-id', b'file-id'])
        work_a.commit('add files')
        self.tree_a = self.workingtree_to_test_tree(work_a)

    def test_path2id(self):
        self.assertEqual(b'bla-id', self.tree_a.path2id('bla'))
        self.assertEqual(b'dir-id', self.tree_a.path2id('dir'))
        self.assertIs(None, self.tree_a.path2id('idontexist'))

    def test_path2id_list(self):
        self.assertEqual(b'bla-id', self.tree_a.path2id(['bla']))
        self.assertEqual(b'dir-id', self.tree_a.path2id(['dir']))
        self.assertEqual(b'file-id', self.tree_a.path2id(['dir', 'file']))
        self.assertEqual(self.tree_a.path2id(''), self.tree_a.path2id([]))
        self.assertIs(None, self.tree_a.path2id(['idontexist']))
        self.assertIs(None, self.tree_a.path2id(['dir', 'idontexist']))

    def test_id2path(self):
        self.addCleanup(self.tree_a.lock_read().unlock)
        self.assertEqual('bla', self.tree_a.id2path(b'bla-id'))
        self.assertEqual('dir', self.tree_a.id2path(b'dir-id'))
        self.assertEqual('dir/file', self.tree_a.id2path(b'file-id'))
        self.assertRaises(errors.NoSuchId, self.tree_a.id2path, b'nonexistant')

    def skip_if_no_reference(self, tree):
        if not getattr(tree, 'supports_tree_reference', lambda: False)():
            raise tests.TestNotApplicable('Tree references not supported')

    def create_nested(self):
        work_tree = self.make_branch_and_tree('wt')
        with work_tree.lock_write():
            self.skip_if_no_reference(work_tree)
            subtree = self.make_branch_and_tree('wt/subtree')
            self.build_tree(['wt/subtree/a'])
            subtree.add(['a'])
            subtree.commit('foo')
            work_tree.add_reference(subtree)
        tree = self._convert_tree(work_tree)
        self.skip_if_no_reference(tree)
        return (tree, subtree)

    def test_path2id_nested_tree(self):
        tree, subtree = self.create_nested()
        self.assertIsNot(None, tree.path2id('subtree'))
        self.assertIsNot(None, tree.path2id('subtree/a'))
        self.assertEqual('subtree', tree.id2path(tree.path2id('subtree')))
        self.assertEqual('subtree/a', tree.id2path(tree.path2id('subtree/a')))
        self.assertIsNot('subtree/a', tree.id2path(tree.path2id('subtree/a'), recurse='down'))
        self.assertRaises(errors.NoSuchId, tree.id2path, tree.path2id('subtree/a'), recurse='none')