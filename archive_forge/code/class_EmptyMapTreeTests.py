from ....tests import TestCase, TestCaseWithTransport
from ....treebuilder import TreeBuilder
from ..maptree import MapTree, map_file_ids
class EmptyMapTreeTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        tree = self.make_branch_and_tree('branch')
        self.oldtree = tree

    def test_has_filename(self):
        self.oldtree.lock_write()
        builder = TreeBuilder()
        builder.start_tree(self.oldtree)
        builder.build(['foo'])
        builder.finish_tree()
        self.maptree = MapTree(self.oldtree, {})
        self.oldtree.unlock()
        self.assertTrue(self.maptree.has_filename('foo'))
        self.assertTrue(self.oldtree.has_filename('foo'))
        self.assertFalse(self.maptree.has_filename('bar'))

    def test_path2id(self):
        self.oldtree.lock_write()
        self.addCleanup(self.oldtree.unlock)
        builder = TreeBuilder()
        builder.start_tree(self.oldtree)
        builder.build(['foo'])
        builder.build(['bar'])
        builder.build(['bla'])
        builder.finish_tree()
        self.maptree = MapTree(self.oldtree, {})
        self.assertEqual(self.oldtree.path2id('foo'), self.maptree.path2id('foo'))

    def test_id2path(self):
        self.oldtree.lock_write()
        self.addCleanup(self.oldtree.unlock)
        builder = TreeBuilder()
        builder.start_tree(self.oldtree)
        builder.build(['foo'])
        builder.build(['bar'])
        builder.build(['bla'])
        builder.finish_tree()
        self.maptree = MapTree(self.oldtree, {})
        self.assertEqual('foo', self.maptree.id2path(self.maptree.path2id('foo')))